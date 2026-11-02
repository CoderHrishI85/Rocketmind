"""Record curated demonstration video clips of RocketMind agents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROCKETMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROCKETMIND_DIR / "src"))

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from rocketmind.utils import get_model_path, get_rocketmind_dir

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


CURATED_RUNS = [
    {
        "agent": "baseline",
        "model_file": "ppo_lunarlander.zip",
        "seed": 1002,  # Known successful landing episode (score 226.19)
        "name_prefix": "baseline_successful_landing",
    },
    {
        "agent": "reward_hack",
        "model_file": "ppo_lunarlander_reward_hack.zip",
        "seed": 1002,  # Stable airborne hovering exploiting time-limit
        "name_prefix": "reward_hack_infinite_hover",
    },
    {
        "agent": "progress_shaped",
        "model_file": "ppo_lunarlander_progress.zip",
        "seed": 1013,  # Controlled landing (score 239.44)
        "name_prefix": "progress_shaped_touchdown",
    },
]


def record_clip(
    model_path: Path,
    seed: int,
    video_dir: Path,
    name_prefix: str,
    device: str = "cpu",
    fps: int = 30,
) -> None:
    """Record a demonstration episode using Pillow GIF fallback or Gymnasium RecordVideo."""
    video_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path), device=device)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []

    try:
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0

        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1

            frame = env.render()
            if frame is not None and steps % 2 == 0:  # Downsample to 25-30 fps
                frames.append(frame)

            if terminated or truncated:
                break

        if frames and HAS_PIL:
            gif_path = video_dir / f"{name_prefix}.gif"
            # Convert numpy arrays to PIL Images
            pil_images = [Image.fromarray(f) for f in frames]
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                optimize=True,
                duration=int(1000 / (fps / 2)),
                loop=0,
            )
            print(f"[MEDIA] Saved demonstration GIF to: {gif_path}")

        print(f"[VIDEO] Recorded '{name_prefix}': Seed={seed}, Steps={steps}, Return={total_reward:.2f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record curated RocketMind demonstration videos")
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["all", "baseline", "reward_hack", "progress_shaped"],
        help="Which agent to record",
    )
    args = parser.parse_args()

    video_dir = get_rocketmind_dir() / "videos"

    for run_spec in CURATED_RUNS:
        if args.agent != "all" and run_spec["agent"] != args.agent:
            continue

        model_path = get_model_path(run_spec["model_file"])
        if not model_path.exists():
            print(f"[WARN] Checkpoint not found for {run_spec['agent']}: {model_path}")
            continue

        print(f"\n[RECORD] Recording demo for {run_spec['agent']}...")
        record_clip(
            model_path=model_path,
            seed=run_spec["seed"],
            video_dir=video_dir,
            name_prefix=run_spec["name_prefix"],
        )

    print(f"\n[SUCCESS] Video recording finished. Clips saved in: {video_dir}")


if __name__ == "__main__":
    main()
