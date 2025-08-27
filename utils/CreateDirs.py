import os
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_project_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config