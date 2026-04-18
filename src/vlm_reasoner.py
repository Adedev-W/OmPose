from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import ConfigError, VLMError
from src.image_io import image_to_data_uri
from src.models.prompts import build_pose_guide_prompt, build_reasoning_prompt, build_repair_prompt
from src.models.schemas import PoseData, VLMReasoningResult, VLMPoseGuideResult


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3-vl-plus"


def extract_json_text(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return cleaned[start : end + 1]
    return cleaned


def parse_reasoning_result(text: str) -> VLMReasoningResult:
    json_text = extract_json_text(text)
    return VLMReasoningResult.model_validate_json(json_text)


def parse_pose_guide_result(text: str) -> VLMPoseGuideResult:
    json_text = extract_json_text(text)
    return VLMPoseGuideResult.model_validate_json(json_text)


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


class VLMReasoner:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._client = None

    def validate_config(self) -> None:
        if not self.api_key:
            raise ConfigError("DASHSCOPE_API_KEY is required")

    @property
    def client(self):
        self.validate_config()
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard.
                raise ConfigError("openai package is required for DashScope OpenAI-compatible calls") from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def recommend(
        self,
        image_path: Path,
        current_pose: PoseData,
        recommendation_count: int = 3,
    ) -> tuple[VLMReasoningResult, dict[str, Any]]:
        prompt = build_reasoning_prompt(current_pose.compact_summary(), recommendation_count)
        data_uri = image_to_data_uri(image_path)
        raw_response, usage = self._complete_with_image(prompt, data_uri)

        try:
            return parse_reasoning_result(raw_response), usage
        except (ValidationError, ValueError) as first_error:
            repaired_response, repair_usage = self._repair_json(raw_response, str(first_error))
            combined_usage = {"initial": usage, "repair": repair_usage}
            try:
                return parse_reasoning_result(repaired_response), combined_usage
            except (ValidationError, ValueError) as second_error:
                raise VLMError(
                    "VLM response could not be parsed as valid OmPose JSON after one repair attempt"
                ) from second_error or first_error

    def select_pose_guide(
        self,
        image_path: Path,
        current_pose: PoseData,
    ) -> tuple[VLMPoseGuideResult, dict[str, Any]]:
        prompt = build_pose_guide_prompt(current_pose.compact_summary())
        data_uri = image_to_data_uri(image_path)
        raw_response, usage = self._complete_with_image(prompt, data_uri)

        try:
            return parse_pose_guide_result(raw_response), usage
        except (ValidationError, ValueError) as first_error:
            repaired_response, repair_usage = self._repair_json(raw_response, str(first_error))
            combined_usage = {"initial": usage, "repair": repair_usage}
            try:
                return parse_pose_guide_result(repaired_response), combined_usage
            except (ValidationError, ValueError) as second_error:
                raise VLMError(
                    "VLM response could not be parsed as valid OmPose pose-guide JSON after one repair attempt"
                ) from second_error or first_error

    def _complete_with_image(self, prompt: str, data_uri: str) -> tuple[str, dict[str, Any]]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:
            raise VLMError(f"DashScope VLM request failed: {exc}") from exc

        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise VLMError("DashScope VLM response was empty")
        return str(content), usage_to_dict(getattr(completion, "usage", None))

    def _repair_json(self, raw_response: str, validation_error: str | None = None) -> tuple[str, dict[str, Any]]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": build_repair_prompt(raw_response, validation_error)}],
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:
            raise VLMError(f"DashScope JSON repair request failed: {exc}") from exc

        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise VLMError("DashScope JSON repair response was empty")
        return str(content), usage_to_dict(getattr(completion, "usage", None))
