from pathlib import Path
import importlib


ROOT = Path(__file__).resolve().parents[1]


def test_required_deployment_files_exist() -> None:
    required = [
        ROOT / "app" / "main.py",
        ROOT / "app" / "config.py",
        ROOT / "app" / "model_manager.py",
        ROOT / "requirements.txt",
        ROOT / "start-backend.txt",
        ROOT / "backend.env.example",
        ROOT / "models" / "MODEL_MANIFEST.json",
        ROOT / "docs" / "README.md",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_deployment_is_cpu_default() -> None:
    from app.config import get_settings

    settings = get_settings()

    assert settings.use_real_omnivoice is False
    assert "models" in str(settings.segmentation_model_path)
    assert "segmentation" in str(settings.segmentation_model_path)
    assert "face" in str(settings.face_detector_path)
    assert "face" in str(settings.face_restorer_path)


def test_model_manager_imports() -> None:
    model_manager = importlib.import_module("app.model_manager")
    status = model_manager.model_status()

    assert len(status) >= 5
    assert {item["feature"] for item in status} >= {
        "background-removal",
        "face-detection",
        "face-restoration",
        "text-to-speech",
    }

