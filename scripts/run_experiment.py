"""Train an RL agent based on an experiment configuration file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path so rocketmind can be imported regardless of execution directory
ROCKETMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROCKETMIND_DIR / "src"))

from stable_baselines3 import PPO

from rocketmind.config import ExperimentConfig
from rocketmind.environments import make_env
from rocketmind.utils import (
    get_rocketmind_dir,
    get_workspace_root,
    save_json,
    set_seed,
)


def train_experiment(
    config: ExperimentConfig,
    override_timesteps: int | None = None,
    smoke_test: bool = False,
) -> Path:
    """Execute training run according to configuration."""
    set_seed(config.seed)

    timesteps = override_timesteps or config.total_timesteps
    if smoke_test:
        timesteps = min(timesteps, 2048)
        print(f"[SMOKE] Smoke test mode: Limiting timesteps to {timesteps}")

    print(f"\n=======================================================")
    print(f"[RUN] Experiment: {config.name}")
    print(f"[INFO] Description: {config.description}")
    print(f"[CONFIG] Wrapper: {config.wrapper_type} | Timesteps: {timesteps:,}")
    print(f"[CONFIG] Seed: {config.seed} | Device: {config.device}")
    print(f"=======================================================\n")

    runs_dir = get_rocketmind_dir() / "runs"
    models_dir = get_workspace_root() / "models"
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_id=config.env_id,
        wrapper_type=config.wrapper_type,
        **config.wrapper_kwargs,
    )

    try:
        model = PPO(
            policy=config.policy,
            env=env,
            verbose=1,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            device=config.device,
            seed=config.seed,
            tensorboard_log=str(runs_dir),
        )

        start_time = time.time()
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=config.tb_log_name,
        )
        duration_sec = round(time.time() - start_time, 2)

        save_path = models_dir / config.model_save_name
        model.save(str(save_path))
        print(f"\n[SUCCESS] Training complete in {duration_sec}s! Model saved: {save_path}.zip")

        # Save metadata alongside model
        meta = config.to_dict()
        meta["duration_seconds"] = duration_sec
        meta["actual_timesteps"] = timesteps
        meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json(meta, save_path.with_suffix(".json"))

        return save_path.with_suffix(".zip")

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RocketMind experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name (e.g. baseline, reward_hack) or path to JSON file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test (2048 steps)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Check configs directory
        potential = ROCKETMIND_DIR / "configs" / f"{args.config}.json"
        if potential.exists():
            cfg_path = potential
        else:
            raise FileNotFoundError(f"Configuration not found: {args.config}")

    config = ExperimentConfig.from_file(cfg_path)
    train_experiment(
        config, override_timesteps=args.timesteps, smoke_test=args.smoke_test
    )


if __name__ == "__main__":
    main()
