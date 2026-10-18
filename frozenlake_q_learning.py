"""Required FrozenLake 8x8 tabular Q-learning hyperparameter experiments.

The Q-learning implementation is adapted from Andrea Pierre's Gymnasium
FrozenLake tutorial (MIT License), then simplified for CPSC 4420 HW2.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gymnasium as gym


# Fixed settings shared by every experiment unless that setting is being tested.
MAP_NAME = "8x8"
IS_SLIPPERY = False
TOTAL_EPISODES = 10_000
N_RUNS = 10
BASE_SEED = 123
ROLLING_WINDOW = 200
FINAL_WINDOW = 500

BASE_ALPHA = 0.5
BASE_GAMMA = 0.90
BASE_EPSILON = 0.10

BEST_ALPHA = 0.10
BEST_GAMMA = 0.99
BEST_EPSILON = 0.05
EVALUATION_EPISODES = 100

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def choose_epsilon_greedy_action(q_table, state, epsilon, rng):
    """Explore with probability epsilon; otherwise choose a best action."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))

    best_actions = np.flatnonzero(q_table[state] == q_table[state].max())
    return int(rng.choice(best_actions))


def train_condition(alpha, gamma, epsilon):
    """Train independent Q-tables and return rewards, steps, and final tables."""
    rewards = np.zeros((N_RUNS, TOTAL_EPISODES), dtype=float)
    steps = np.zeros((N_RUNS, TOTAL_EPISODES), dtype=float)
    q_tables = []

    for run in range(N_RUNS):
        run_seed = BASE_SEED + run
        rng = np.random.default_rng(run_seed)
        env = gym.make(
            "FrozenLake-v1",
            map_name=MAP_NAME,
            is_slippery=IS_SLIPPERY,
        )
        env.action_space.seed(run_seed)

        state_count = env.observation_space.n
        action_count = env.action_space.n
        q_table = np.zeros((state_count, action_count), dtype=float)

        for episode in range(TOTAL_EPISODES):
            state, _ = env.reset(seed=run_seed if episode == 0 else None)
            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done:
                action = choose_epsilon_greedy_action(
                    q_table,
                    state,
                    epsilon,
                    rng,
                )
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Terminal states have no future value.
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.max(q_table[new_state])

                prediction_error = target - q_table[state, action]
                q_table[state, action] += alpha * prediction_error

                state = new_state
                episode_reward += reward
                episode_steps += 1

            rewards[run, episode] = episode_reward
            steps[run, episode] = episode_steps

        q_tables.append(q_table.copy())
        env.close()

    return rewards, steps, np.asarray(q_tables)


def rolling_mean(values, window):
    """Return a same-length rolling mean with a growing initial window."""
    cumulative = np.cumsum(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        result[index] = total / (index - start + 1)
    return result


def summarize_condition(experiment, condition, alpha, gamma, epsilon, rewards, steps):
    """Compute the three comparisons requested by the assignment."""
    mean_curve = rewards.mean(axis=0)
    smoothed_curve = rolling_mean(mean_curve, ROLLING_WINDOW)
    threshold_matches = np.flatnonzero(smoothed_curve >= 0.50)
    episode_to_50_percent = (
        int(threshold_matches[0] + 1) if len(threshold_matches) else np.nan
    )

    per_run_final_rewards = rewards[:, -FINAL_WINDOW:].mean(axis=1)
    final_steps = steps[:, -FINAL_WINDOW:]
    successful_steps = final_steps[rewards[:, -FINAL_WINDOW:] > 0]

    return {
        "experiment": experiment,
        "condition": condition,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "episodes": TOTAL_EPISODES,
        "runs": N_RUNS,
        "episode_to_50_percent_success": episode_to_50_percent,
        "final_mean_reward": float(per_run_final_rewards.mean()),
        "final_reward_stability_std_across_runs": float(
            per_run_final_rewards.std(ddof=1)
        ),
        "final_mean_steps_on_success": (
            float(successful_steps.mean()) if successful_steps.size else np.nan
        ),
    }


def plot_experiment(title, fixed_conditions, curves, output_path):
    """Plot rolling reward with variability across independent runs."""
    figure, axis = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, TOTAL_EPISODES + 1)

    for label, rewards in curves.items():
        rolled_runs = np.vstack(
            [rolling_mean(run_rewards, ROLLING_WINDOW) for run_rewards in rewards]
        )
        mean_curve = rolled_runs.mean(axis=0)
        std_curve = rolled_runs.std(axis=0)
        axis.plot(episodes, mean_curve, linewidth=2, label=label)
        axis.fill_between(
            episodes,
            np.clip(mean_curve - std_curve, 0, 1),
            np.clip(mean_curve + std_curve, 0, 1),
            alpha=0.14,
        )

    axis.set_title(f"{title}\n{fixed_conditions}")
    axis.set_xlabel("Training episode")
    axis.set_ylabel(f"Reward / success rate ({ROLLING_WINDOW}-episode mean)")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(title="Tested condition")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def select_best_training_run(rewards, q_tables):
    """Return the strongest run based on its final training window."""
    final_scores = rewards[:, -FINAL_WINDOW:].mean(axis=1)
    index = int(np.argmax(final_scores))
    return index, float(final_scores[index]), q_tables[index]


def evaluate_policy(q_table, seed_offset=30_000):
    """Evaluate a learned policy with exploration disabled."""
    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
    )
    rewards = []
    steps = []

    for episode in range(EVALUATION_EPISODES):
        state, _ = env.reset(seed=BASE_SEED + seed_offset + episode)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)

    env.close()
    rewards = np.asarray(rewards)
    steps = np.asarray(steps)
    successful_steps = steps[rewards > 0]

    return {
        "evaluation_episodes": EVALUATION_EPISODES,
        "evaluation_epsilon": 0.0,
        "mean_reward": float(rewards.mean()),
        "success_rate": float((rewards > 0).mean()),
        "mean_steps_all_episodes": float(steps.mean()),
        "mean_steps_successful_episodes": (
            float(successful_steps.mean()) if successful_steps.size else np.nan
        ),
    }


def save_training_plot(rewards, title, color, filename):
    """Save a single configuration's mean learning curve."""
    rolled_runs = np.vstack(
        [rolling_mean(run_rewards, ROLLING_WINDOW) for run_rewards in rewards]
    )
    mean_curve = rolled_runs.mean(axis=0)
    std_curve = rolled_runs.std(axis=0)
    episodes = np.arange(1, TOTAL_EPISODES + 1)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(episodes, mean_curve, linewidth=2, color=color)
    axis.fill_between(
        episodes,
        np.clip(mean_curve - std_curve, 0, 1),
        np.clip(mean_curve + std_curve, 0, 1),
        alpha=0.18,
        color=color,
    )
    axis.set_title(title)
    axis.set_xlabel("Training episode")
    axis.set_ylabel(f"Reward / success rate ({ROLLING_WINDOW}-episode mean)")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_comparison_plot(comparison):
    """Plot the baseline and best-policy evaluation metrics."""
    labels = comparison["agent"].tolist()
    colors = ["#7289a8", "#2d7f5e"]
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4.5))

    axes[0].bar(labels, comparison["mean_reward"], color=colors)
    axes[0].set_title("Mean reward")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(labels, comparison["success_rate"], color=colors)
    axes[1].set_title("Success rate")
    axes[1].set_ylim(0, 1.05)
    axes[2].bar(
        labels,
        comparison["mean_steps_successful_episodes"],
        color=colors,
    )
    axes[2].set_title("Mean successful steps")
    axes[2].set_ylim(
        0,
        max(comparison["mean_steps_successful_episodes"]) * 1.15,
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)

    figure.suptitle(
        "FrozenLake 8x8 baseline versus best policy\n"
        "100 evaluation episodes, epsilon=0"
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "baseline_vs_best_evaluation.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_experiment(experiment, values, varying_parameter):
    """Run one three-condition sweep while holding all other settings fixed."""
    curves = {}
    summaries = []

    for value in values:
        alpha = value if varying_parameter == "alpha" else BASE_ALPHA
        gamma = value if varying_parameter == "gamma" else BASE_GAMMA
        epsilon = value if varying_parameter == "epsilon" else BASE_EPSILON

        condition = (
            f"alpha={alpha:.2f}, gamma={gamma:.2f}, epsilon={epsilon:.2f}"
        )
        print(f"Running {experiment}: {condition}", flush=True)
        rewards, steps, _ = train_condition(alpha, gamma, epsilon)
        curves[condition] = rewards
        summaries.append(
            summarize_condition(
                experiment,
                condition,
                alpha,
                gamma,
                epsilon,
                rewards,
                steps,
            )
        )

    if varying_parameter == "alpha":
        filename = (
            "learning_rate_experiment_"
            "alpha_0.10_0.50_0.90_gamma_0.90_epsilon_0.10.png"
        )
        title = "Learning-rate experiment on FrozenLake 8x8"
        fixed = "Fixed: gamma=0.90, epsilon=0.10"
    elif varying_parameter == "gamma":
        filename = (
            "discount_factor_experiment_"
            "gamma_0.50_0.90_0.99_alpha_0.50_epsilon_0.10.png"
        )
        title = "Discount-factor experiment on FrozenLake 8x8"
        fixed = "Fixed: alpha=0.50, epsilon=0.10"
    else:
        filename = (
            "epsilon_experiment_"
            "epsilon_0.05_0.10_0.30_alpha_0.50_gamma_0.90.png"
        )
        title = "Exploration-rate experiment on FrozenLake 8x8"
        fixed = "Fixed: alpha=0.50, gamma=0.90"

    plot_experiment(title, fixed, curves, RESULTS_DIR / filename)
    return summaries


def run_required_experiments():
    """Run every required experiment and best-agent evaluation."""
    all_summaries = []
    all_summaries.extend(
        run_experiment(
            "Learning-rate experiment",
            [0.10, 0.50, 0.90],
            "alpha",
        )
    )
    all_summaries.extend(
        run_experiment(
            "Discount-factor experiment",
            [0.50, 0.90, 0.99],
            "gamma",
        )
    )
    all_summaries.extend(
        run_experiment(
            "Exploration-rate experiment",
            [0.05, 0.10, 0.30],
            "epsilon",
        )
    )

    summary_frame = pd.DataFrame(all_summaries)
    print("\nExperiment summary:")
    print(summary_frame.to_string(index=False))

    print("\nTraining baseline configuration", flush=True)
    baseline_rewards, _, baseline_tables = train_condition(
        BASE_ALPHA,
        BASE_GAMMA,
        BASE_EPSILON,
    )
    baseline_run, baseline_score, baseline_q_table = select_best_training_run(
        baseline_rewards,
        baseline_tables,
    )
    save_training_plot(
        baseline_rewards,
        "Baseline training on FrozenLake 8x8\n"
        "alpha=0.50, gamma=0.90, epsilon=0.10",
        "#7289a8",
        "baseline_training_alpha_0.50_gamma_0.90_epsilon_0.10.png",
    )

    print("Training best combined configuration", flush=True)
    best_rewards, _, best_tables = train_condition(
        BEST_ALPHA,
        BEST_GAMMA,
        BEST_EPSILON,
    )
    best_run, best_score, best_q_table = select_best_training_run(
        best_rewards,
        best_tables,
    )
    save_training_plot(
        best_rewards,
        "Best combined configuration on FrozenLake 8x8\n"
        "alpha=0.10, gamma=0.99, epsilon=0.05",
        "#2d7f5e",
        "best_agent_training_alpha_0.10_gamma_0.99_epsilon_0.05.png",
    )

    baseline_metrics = evaluate_policy(baseline_q_table)
    best_metrics = evaluate_policy(best_q_table)
    comparison = pd.DataFrame(
        [
            {
                "agent": "Baseline",
                "alpha": BASE_ALPHA,
                "gamma": BASE_GAMMA,
                "epsilon": BASE_EPSILON,
                "selected_training_run": baseline_run,
                "selected_run_final_training_reward": baseline_score,
                **baseline_metrics,
            },
            {
                "agent": "Best",
                "alpha": BEST_ALPHA,
                "gamma": BEST_GAMMA,
                "epsilon": BEST_EPSILON,
                "selected_training_run": best_run,
                "selected_run_final_training_reward": best_score,
                **best_metrics,
            },
        ]
    )
    save_comparison_plot(comparison)

    print("\nBaseline versus best evaluation:")
    print(comparison.to_string(index=False))
    print(f"\nSaved required plots to: {RESULTS_DIR}")


def demonstrate_best_agent():
    """Train the best configuration and display one episode for recording."""
    print(
        "Training the best configuration before the demonstration...",
        flush=True,
    )
    rewards, _, q_tables = train_condition(
        BEST_ALPHA,
        BEST_GAMMA,
        BEST_EPSILON,
    )
    run_index, training_score, q_table = select_best_training_run(
        rewards,
        q_tables,
    )

    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode="human",
    )
    state, _ = env.reset(seed=BASE_SEED + 20_000)
    env.render()

    print(
        f"Selected training run {run_index} "
        f"(final reward={training_score:.3f})."
    )
    input(
        "The FrozenLake window is open. Start your screen recording, "
        "then press Enter here to begin the agent."
    )
    time.sleep(1.0)

    done = False
    total_reward = 0.0
    step_count = 0
    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        time.sleep(0.5)

    print(f"Finished with reward={total_reward:.0f} in {step_count} steps.")
    print("Stop your screen recording now.")
    time.sleep(3.0)
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="FrozenLake 8x8 tabular Q-learning assignment"
    )
    parser.add_argument(
        "--demo-best",
        action="store_true",
        help="open and run the best agent for manual screen recording",
    )
    args = parser.parse_args()

    if args.demo_best:
        demonstrate_best_agent()
    else:
        run_required_experiments()


if __name__ == "__main__":
    main()
