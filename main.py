
import gymnasium as gym


def main():
    # Launch the Lunar Lander simulation.
    env = gym.make("LunarLander-v3", render_mode="human")

    try:
        for episode in range(1, 4):
            observation, info = env.reset(seed=episode)

            total_reward = 0.0
            steps = 0

            while True:
                # Our first agent acts randomly.
                action = env.action_space.sample()

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    print(
                        f"Episode {episode} | "
                        f"Steps: {steps} | "
                        f"Reward: {total_reward:.2f}"
                    )
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()
