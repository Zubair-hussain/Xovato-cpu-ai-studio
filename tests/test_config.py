from app.config import Settings


def test_cpu_defaults():
    settings = Settings()
    assert settings.use_real_omnivoice is False
    assert settings.omnivoice_quantize_int8 is True
    assert "segmentation" in str(settings.segmentation_model_path)


def test_cors_origin_list_splits_and_trims():
    settings = Settings(cors_origins="https://a.com, https://b.com ,")
    assert settings.cors_origin_list == ["https://a.com", "https://b.com"]


def test_cors_wildcard_passthrough():
    settings = Settings(cors_origins="*")
    assert settings.cors_origin_list == ["*"]


def test_limits_are_positive():
    settings = Settings()
    assert settings.max_image_upload_bytes > 0
    assert settings.max_voice_sample_bytes > 0
    assert settings.max_shorts_clips == 20
    assert settings.max_shorts_jobs >= 1
