from __future__ import annotations


class OmPoseError(Exception):
    exit_code = 99
    label = "OmPose error"


class ConfigError(OmPoseError):
    exit_code = 1
    label = "Config error"


class ImageInputError(OmPoseError):
    exit_code = 2
    label = "Image input error"


class NoPoseDetectedError(ImageInputError):
    label = "No pose detected"


class VLMError(OmPoseError):
    exit_code = 3
    label = "VLM error"


class OverlaySaveError(OmPoseError):
    exit_code = 4
    label = "Overlay save error"

