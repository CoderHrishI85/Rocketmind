"""Experiment configuration schemas and loaders for RocketMind.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    name: str
    description: str
    algorithm: str = "PPO"
    env_id: str = "LunarLander-v3"
    wrapper_type: str = "none"
    wrapper_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    total_timesteps: int = 300_000
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    device: str = "cpu"
    tb_log_name: str = "PPO_Run"
    model_save_name: str = "ppo_model"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

