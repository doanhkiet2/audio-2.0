import os
from pathlib import Path
import yaml


def load_config():
    env = os.getenv("APP_ENV", "local")

    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "config" / f"{env}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg.get("project_root", "."))
    if not project_root.is_absolute():
        project_root = base_dir / project_root

    cfg["project_root"] = project_root
    cfg["input_dir"] = Path(cfg["input_dir"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["log_file"] = Path(cfg["log_file"])

    if not cfg["input_dir"].is_absolute():
        cfg["input_dir"] = project_root / cfg["input_dir"]

    if not cfg["output_dir"].is_absolute():
        cfg["output_dir"] = project_root / cfg["output_dir"]

    if not cfg["log_file"].is_absolute():
        cfg["log_file"] = project_root / cfg["log_file"]

    return cfg
