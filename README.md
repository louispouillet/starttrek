# Start Trek: Lunar Lander

Two reinforcement learning agents learn to land the lunar module of Gymnasium's `LunarLander-v3` by trial and error: a **DQN** agent and a **PPO** agent. The environment is solved when an agent reaches a mean score of at least 200 over 100 consecutive episodes, played without exploration.

## Repository

| File | Content |
|---|---|
| `dqn.ipynb` | the DQN agent: replay buffer, epsilon greedy, target network, training, results and experiments |
| `ppo.ipynb` | the PPO agent: actor and critic, rollout storage, GAE, clipped loss, training, results, experiment, comparison with DQN and conclusion |
| `dqn_lunarlander.pt` | the saved DQN model |
| `ppo_lunarlander.pt` | the saved PPO model |
| `model.py` | loads a saved model and shows it playing in a window |
| `DQN.md` | a short summary of the DQN agent and its results |
| `PPO.md` | a short summary of the PPO agent and its results |
| `requirements.txt` | the pinned dependencies |
| `StartTrek.pdf` | the subject |

## Installation

Python 3.14 is used.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks

```bash
jupyter lab
```

Open a notebook and run *Restart & Run All*: the cells run from top to bottom.

Each notebook has a `RETRAIN` switch. In `ppo.ipynb`, `RETRAIN = False` reloads a saved training (model and training log) when it was made with the same configuration, so the whole notebook runs in a few minutes. When the files are missing, the agent is trained and the files are written: one PPO training takes about 5 minutes on a laptop CPU, and the notebook trains 6 of them (3 seeds and the experiment of section 9). With the same seed, a training gives the same agent.

In `dqn.ipynb`, `RETRAIN = True`: the notebook trains the baseline agent and every configuration of its experiments again, which took about 3 hours on the computer of its author. The experiment of section 7 writes the final model `dqn_lunarlander.pt`.

Only the final models are in the repository: the training logs and the models of the other seeds are written locally by the notebooks.

## Watching a saved agent

```bash
python model.py ALGO MODEL [EPISODES]
```

| Argument | Meaning |
|---|---|
| `ALGO` | algorithm of the saved model: `dqn` or `ppo` |
| `MODEL` | path to the saved model file |
| `EPISODES` | number of episodes to run (default: 1) |

```bash
python model.py -h
python model.py ppo ppo_lunarlander.pt 3
```

The agent plays without exploration in a display window, and the score of each episode is printed:

```
Episode 1: 261.1
Episode 2: 301.9
```

`model.py` exits with code 0 on success. On an error (wrong number of arguments, unknown algorithm, missing or invalid model file, invalid number of episodes), it writes the message on the error output and exits with code 84.

## Results

Evaluation over 100 consecutive episodes without exploration.

| Agent | Mean ± std (same 100 episodes) | Episodes above 200 | Three seeds | Solved |
|---|---|---|---|---|
| random | −177.2 ± 94.6 | 0 | | no |
| PPO | 262.2 ± 46.7 | 92 | 254.3 ± 13.9 | yes |
| DQN | 246.8 ± 53.4 | 94 | 251.0 ± 9.6 | yes |

Both agents solve the environment at the same level. DQN learns in fewer episodes and in less time, PPO has the higher typical score. Details: `ppo.ipynb` (section 10 for the comparison) and `PPO.md` for PPO, `dqn.ipynb` and `DQN.md` for DQN.
