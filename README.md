# CPSC 4420 Assignment 2 - FrozenLake Q-Learning

This project implements tabular Q-learning for Gymnasium's standard 8x8
FrozenLake environment. It measures how the learning rate (`alpha`), discount
factor (`gamma`), and exploration rate (`epsilon`) affect learning speed,
stability, and final training performance.

## Environment

- Environment: `FrozenLake-v1`
- Map: standard 8x8 map
- Transition setting: deterministic (`is_slippery=False`)
- States: 64 discrete grid locations
- Actions: left, down, right, and up
- Reward: 1 for reaching the goal and 0 otherwise
- Episode end: reaching the goal, entering a hole, or reaching the time limit

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run the experiments

```bash
python frozenlake_q_learning.py
```

This runs three values for each required hyperparameter and writes labeled
figures to `results/`.

The same script also trains and evaluates the baseline and best combined
configuration over 100 episodes with exploration disabled.

## Manually record the best agent

```bash
python frozenlake_q_learning.py --demo-best
```

The script trains the best configuration and opens a live FrozenLake window.
When prompted, start macOS screen recording with `Shift + Command + 5`, then
press Enter in Terminal. Stop the recording after the agent reaches the goal.
The script itself does not record or export a video.

## Tested conditions

- Learning rate: `alpha = 0.10, 0.50, 0.90`
- Discount factor: `gamma = 0.50, 0.90, 0.99`
- Exploration rate: `epsilon = 0.05, 0.10, 0.30`
- Episodes per run: 10,000
- Independent runs per condition: 10

The baseline uses `alpha=0.50`, `gamma=0.90`, and `epsilon=0.10`. The combined
best candidate uses `alpha=0.10`, `gamma=0.99`, and `epsilon=0.05`.

## Main results

- Learning-rate conditions converged at similar speeds. `alpha=0.10` had the
  highest final training reward, while `alpha=0.90` varied least across runs.
- `gamma=0.99` had the highest final reward in the discount-factor comparison.
- `epsilon=0.05` produced the strongest final training performance.
- `epsilon=0.30` continued making many random moves and substantially reduced
  final training performance.
- With exploration disabled, both the baseline and best policies achieved a
  100% success rate and completed the deterministic map in 14 steps.

## Project structure

```text
frozenlake_q_learning.py       Main hyperparameter experiments
results/                       Required experiment and evaluation figures
videos/                        Manually recorded best-policy MP4
requirements.txt               Python dependencies
CPSC4420_HW2_Report.pdf         Report submitted to Canvas
```

## Attribution

The initial Q-learning structure was adapted from Andrea Pierre's MIT-licensed
Gymnasium FrozenLake tutorial and then simplified and extended for this
assignment.

