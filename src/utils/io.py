import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ISO_FORMAT = "%d-%m-%Y-%H-%M"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def timestamp() -> str:
    return datetime.now().strftime(ISO_FORMAT)

def dump_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
