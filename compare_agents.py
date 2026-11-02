
from pathlib import Path
import csv
import statistics

import gymnasium as gym
from stable_baselines3 import PPO


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

MODELS = {
    "baseline": ROOT_DIR / "models" / "ppo_lunarlander.zip",

    "reward_hack": ROOT_DIR / "models" / "ppo_lunarlander_reward_hack.zip",

    "progress_shaped": ROOT_DIR / "models" / "ppo_lunarlander_progress.zip",
}

N_EPISODES = 20


def run_evaluation(env, model, agent_name, episode, seed):
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    stable_hover_frames = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        # Measure the flawed reward's target behavior,
        # but do NOT add this bonus during evaluation.
        stable_hover = (
            abs(obs[2]) < 0.2
            and abs(obs[3]) < 0.2
            and abs(obs[4]) < 0.15
            and abs(obs[5]) < 0.2
            and obs[6] < 0.5
            and obs[7] < 0.5
            and not terminated
            and not truncated
        )

        if stable_hover:
            stable_hover_frames += 1

        if terminated or truncated:
            break

    return {
        "agent": agent_name,
        "episode": episode,
        "seed": seed,
        "reward": round(total_reward, 3),
        "steps": steps,
        "score_200_plus": total_reward >= 200,
        "truncated": truncated,
        "stable_hover_frames": stable_hover_frames,
    }


def main():
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} model: {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Both agents get the same environment settings and seeds.
    env = gym.make("LunarLander-v3")

    try:
        for agent_name, model_path in MODELS.items():
            print(f"\nEvaluating: {agent_name}")
            model = PPO.load(str(model_path), device="cpu")
            agent_results = []

            for episode in range(1, N_EPISODES + 1):
                seed = 1000 + episode

                result = run_evaluation(
                    env, model, agent_name, episode, seed
                )
                agent_results.append(result)
                all_results.append(result)

                print(
                    f"{episode:02d}/{N_EPISODES} | "
                    f"Reward: {result['reward']:8.2f} | "
                    f"Steps: {result['steps']:4d} | "
                    f"Hover frames: {result['stable_hover_frames']:4d}"
                )

            rewards = [r["reward"] for r in agent_results]
            print(f"\n{agent_name} summary")
            print(f"Mean raw reward: {statistics.mean(rewards):.2f}")
            print(f"Median reward: {statistics.median(rewards):.2f}")
            print(
                "Scores >= 200: "
                f"{sum(r['score_200_plus'] for r in agent_results)}/{N_EPISODES}"
            )
            print(
                "Time-limit truncations: "
                f"{sum(r['truncated'] for r in agent_results)}/{N_EPISODES}"
            )
            print(
                "Mean stable-hover frames: "
                f"{statistics.mean(r['stable_hover_frames'] for r in agent_results):.1f}"
            )

    finally:
        env.close()

    output_path = RESULTS_DIR / "reward_hack_vs_baseline.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
