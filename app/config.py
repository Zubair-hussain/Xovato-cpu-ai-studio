from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


class Settings(BaseSettings):
    app_name: str = "EnhanceAI Core Backend"
    environment: str = "local"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    data_dir: Path = BASE_DIR / "data"
    log_dir: Path = BASE_DIR / "data" / "logs"
    omnivoice_source_dir: Path = BASE_DIR / "vendor" / "OmniVoice-Studio"
    omnivoice_model_dir: Path = MODEL_DIR / "omnivoice" / "k2-fsa_OmniVoice"
    use_real_omnivoice: bool = False
    # Speed/quality levers for CPU-only synthesis. Fewer diffusion steps and a
    # guidance scale of 1.0 (which skips the second, unconditional forward pass)
    # are the two biggest levers after int8 quantisation.
    omnivoice_quantize_int8: bool = True
    omnivoice_num_step: int = 8
    omnivoice_guidance_scale: float = 1.6
    # Small U^2-Net (~4.5 MB) used for CPU-only foreground segmentation.
    segmentation_model_path: Path = MODEL_DIR / "segmentation" / "u2netp.onnx"
    max_image_upload_bytes: int = 32 * 1024 * 1024
    max_audio_upload_bytes: int = 128 * 1024 * 1024
    max_shorts_upload_bytes: int = 2 * 1024 * 1024 * 1024
    # YuNet face detector (~230 KB) and GPEN-BFR-256 restorer (~72 MB), CPU-only.
    face_detector_path: Path = MODEL_DIR / "face" / "yunet.onnx"
    face_restorer_path: Path = MODEL_DIR / "face" / "gpen_bfr_256.onnx"
    local_cpu_warning_percent: float = 85.0
    local_cpu_critical_percent: float = 95.0
    omnivoice_breaker_recovery_seconds: int = 1800
    omnivoice_breaker_heavy_recovery_seconds: int = 3600
    segmentation_breaker_recovery_seconds: int = 900

    model_config = SettingsConfigDict(env_prefix="ENHANCEAI_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
