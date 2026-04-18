from __future__ import annotations

import json
from typing import Any

from src.models.schemas import PoseTemplateId


POSE_TEMPLATE_IDS: tuple[PoseTemplateId, ...] = (
    "seated_relaxed",
    "seated_open_shoulders",
    "seated_hand_on_thigh",
    "seated_side_lean",
    "standing_relaxed",
    "standing_contrapposto",
    "hand_on_hip",
    "arms_open_soft",
    "walking_stride_soft",
    "leaning_side",
    "portrait_chin_angle",
    "one_hand_near_face",
)


def build_pose_guide_prompt(pose_summary: dict[str, Any]) -> str:
    pose_json = json.dumps(pose_summary, ensure_ascii=False, indent=2)
    template_ids = ", ".join(POSE_TEMPLATE_IDS)
    return f"""
You are a professional camera pose director.

Analyze the attached user photo and detected pose summary. Choose one practical pose guide that improves the shot while staying realistic for the person's current framing, space, and body position.

Important: you are a selector only. Do not create coordinates, skeletons, or target_keypoints. A local geometry engine will generate the overlay.

Return JSON only. Do not wrap the JSON in markdown fences. Do not include commentary outside JSON.

Detected pose summary:
{pose_json}

Available pose_template_id values:
{template_ids}

Return exactly this JSON shape:
{{
  "scene": {{
    "scene_type": "indoor|outdoor|mixed|unknown",
    "location_category": "beach/park/cafe/studio/street/mountain/urban/...",
    "lighting": "natural/artificial/golden_hour/backlit/low_light/...",
    "mood": "casual/formal/romantic/adventurous/professional/...",
    "space_constraints": {{
      "available_width": "narrow|medium|wide",
      "available_height": "low_ceiling|medium|open_sky",
      "ground_type": "flat|stairs|uneven|seated_available|unknown"
    }},
    "key_elements": ["short visual elements in the scene"],
    "composition_notes": "brief scene-specific note"
  }},
  "recommendation": {{
    "pose_template_id": "one value from the available list",
    "name": "Short pose name",
    "category": "standing|sitting|leaning|walking|dramatic|casual",
    "description": "One short actionable instruction for the user",
    "reasoning": "Why this pose fits this specific scene and framing",
    "difficulty": "easy|medium|hard",
    "camera_angle_suggestion": "eye-level/low-angle/high-angle/...",
    "guide_params": {{
      "lean": -0.4
    }},
    "correction_focus": ["short body parts to coach, e.g. shoulders", "right wrist"]
  }},
  "usage_notes": "brief note, optional"
}}

Rules:
- Return exactly one recommendation.
- Never output target_keypoints, x/y coordinates, pixels, or skeleton geometry.
- If the person is seated, close to camera, or lower body is cropped, choose a seated or portrait-safe template.
- Keep the pose easy unless the scene clearly supports a more expressive full-body guide.
- guide_params must be small, optional steering values only; use "lean" from -1.0 to 1.0 when useful.
- Keep descriptions concise enough for a camera UI.
""".strip()


def build_reasoning_prompt(pose_summary: dict[str, Any], recommendation_count: int = 1) -> str:
    return build_pose_guide_prompt(pose_summary)


def build_repair_prompt(raw_response: str, validation_error: str | None = None) -> str:
    error_section = f"\nValidation error to fix:\n{validation_error}\n" if validation_error else ""
    return f"""
Convert the following model response into valid JSON matching the OmPose schema. Return JSON only, no markdown fences, no explanation.

Rules:
- Preserve the scene and recommendation intent when possible.
- Output exactly one "recommendation" object.
- recommendation.pose_template_id must be one of: {", ".join(POSE_TEMPLATE_IDS)}.
- Do not include target_keypoints, coordinates, pixels, or skeleton geometry.
- Use guide_params only for small selector hints such as "lean".
{error_section}

Raw response:
{raw_response}
""".strip()
