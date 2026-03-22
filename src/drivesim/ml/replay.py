from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class ReplayLogger:
    def __init__(self, path: str = "replays/latest_episode.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, payload: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
