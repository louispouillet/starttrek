# DQN on LunarLander-v3

A summary of the DQN agent of `dqn.ipynb`, to read in a few minutes. Every number comes from an output of the notebook (except the comparison with PPO, which comes from `ppo.ipynb`), and each part points to the section that proves it.

## Result

| Measure | Value |
|---|---|
| score of the final model, seed 42, over 100 episodes without exploration | **249.8** (solved: threshold 200) |
| three seeds (42, 43, 44) | **251.0 ± 9.6** (249.8, 263.3, 240.0), all above 200 |
| mean std inside the 100-episode evaluations, over the seeds | 59.3 |
| moving average reaches 200 (seeds 42, 43, 44) | episodes 315, 328, 459 |
| training | 600 episodes |
| baseline model, over 100 episodes | 235.1 ± 99.2, 90 / 100 above 200 (221.0 ± 113.2, 82 / 100 on other episodes right after the training) |
| random agent, for reference | -173.4 ± 110.0, 0 / 100 above 200 |

The final model is `dqn_lunarlander.pt`. The baseline model is `dqn_lunarlander_baseline.pt` (not in the repository).

## How DQN learns

```
┌─► 1. PLAY     one step: the agent picks an action (random with probability epsilon, otherwise the best Q-value)
│   2. STORE    the experience (state, action, reward, next state, done) goes into the replay buffer
│   3. LEARN    a batch of 64 random experiences: the network moves towards the Bellman target
└── 4. COPY     every 500 steps, the target network becomes a copy of the network
```

DQN is **off policy**: the stored experiences can be used again and again, and the network learns at every step (no rollout thrown away, unlike PPO).

| Element | Meaning |
|---|---|
| **Q-network** | network that gives one value per action: the reward expected if this action is played, then the best ones |
| **replay buffer** | the last 100,000 experiences; a random batch breaks the similarity between consecutive steps |
| **Bellman target** | $r + \gamma \max_a Q_{target}(s', a)\,(1 - done)$, computed without gradient: a constant |
| **target network** | copy of the network, refreshed every 500 steps, so that the target does not move at every update |
| **loss** | MSE between the prediction $Q(s, a)$ for the action played and the Bellman target |
| **best checkpoint** | evaluated without exploration every 25 episodes; the best weights are loaded back at the end |

## Exploration

The agent plays a random action with probability epsilon, otherwise the action with the highest Q-value. Epsilon decays **once per episode**, exponentially: $\epsilon = 1.0 \times \text{eps\_decay}^{episode}$, with a floor of 0.01. During evaluation and in `model.py`, epsilon is 0 (the greedy action), as the subject requires.

With the baseline decay (0.995), epsilon is still about 0.05 at episode 600. With the final decay (0.98), epsilon is below 0.1 from episode 115. This is the setting that changed the results the most (see the experiments).

## `terminated` and `truncated`

| Flag | Meaning | Treatment |
|---|---|---|
| `terminated` | real end: crash, out of the viewport, at rest | `done = 1`: no future in the target |
| `truncated` | limit of 1000 steps: the lander could have kept going | `done = 0`: the future is still counted in the target |

Only `terminated` is stored as `done`. The end reason of each episode (`crash`, `out_of_viewport`, `at_rest`, `truncated`) is recorded during training and during every evaluation. A crash is tested first, so a crash that throws the lander off screen counts as a crash.

## Architecture (notebook section 4)

```
state (8) ─ Linear(8, 128) ─ ReLU ─ Linear(128, 128) ─ ReLU ─ Linear(128, 4) ─► Q-values (4)     (18,180 parameters)
```

* **One network, one Q-value per action**: a single forward pass gives the value of the four actions, and the agent plays the highest one.
* **ReLU**: without it, stacking `Linear` layers would be equivalent to a single one.
* **No activation at the output**: a Q-value can be negative, and it is read as is.
* The target network has the same architecture and is only copied, never trained.

## Hyperparameters

| Field | Value | Reason |
|---|---|---|
| `gamma` | 0.99 | the landing comes at the end of a long episode, the future must count almost as much as the present |
| `learning_rate` | 1e-3 | usual value for a small network, tested in experiment 2 (kept) |
| `batch_size` | 64 | usual value for a small network |
| `hidden_size` | 128 | usual value for 8 inputs |
| `buffer_capacity` | 100,000 | a training plays about 200,000 steps: the buffer keeps about the latest half |
| `eps_start`, `eps_end` | 1.0, 0.01 | start fully random, floor of 1% |
| `eps_decay` | **0.98** (baseline: 0.995) | chosen by experiment 1 |
| `target_update_every` | 500 | often enough to follow the network, rarely enough for the target not to move at every update |
| `n_episodes` | 600 | enough to start landing, short enough to repeat the training in the experiments |
| `eval_every`, `eval_episodes` | 25, 10 | one evaluation without exploration every 25 episodes, on 10 episodes, to choose the best checkpoint |

## Training of the baseline (notebook sections 5 and 6)

* The moving average over 100 episodes goes from -138 at episode 100 to 161 at episode 600. Its maximum is 199 (episode 566): with exploration still around 5%, **the training reward never reaches 200**.
* The evaluations without exploration go from -606 to +266, and only **3 out of 24 are above 200**. They fall back to -65 at episode 500 and -258 at episode 575, and the last weights score 130. This is why the best checkpoint is kept (here episode 475, 266.4 on 10 episodes).
* The best checkpoint is chosen on 10 episodes, then measured again on **100 new episodes**: 235.1 ± 99.2, with 90 above 200. The 31 points between 266.4 and 235.1 are the cost of choosing the best of 24 noisy evaluations.
* End reasons during the baseline training: 333 crashes, 180 at rest, 73 truncated, 14 out of the viewport. The 73 truncated episodes mostly happen between episodes 300 and 400: the agent learns to **hover** before it learns to land (5 truncated in the last 100 episodes).
* End reasons of the 100 evaluation episodes: 94 at rest, 2 truncated, 4 crashes, 0 out of the viewport; 267 steps on average, against 88 for the random agent.
* The loss goes from about 57 to 18 while the evaluation oscillates: it is not a measure of quality, because the target moves with the network.

## Experiments over several seeds (notebook section 7)

Each configuration is trained on three seeds (42, 43, 44), one parameter changes at a time, and experiment 2 starts from the winner of experiment 1. A cache avoids training the same configuration twice.

### Experiment 1: epsilon decay

| `eps_decay` | 100-episode score per seed | Mean ± spread | Moving average reaches 200 |
|---|---|---|---|
| 0.995 (baseline) | 244.9, 225.0, 215.3 | 228.4 ± 12.4 | 531, 523, never |
| 0.99 | -127.8, 265.4, 231.3 | 123.0 ± 177.9 | never, 283, 466 |
| **0.98** | 249.8, 263.3, 240.0 | **251.0 ± 9.6** | 315, 328, 459 |

* 0.98 wins: +23 points on average, the smallest spread, and 200 reached on all seeds, about 200 episodes earlier than the baseline on the two seeds where the baseline gets there.
* 0.99 is unstable: one seed ends at -127.8 and never learns. The cause was not investigated.
* A first version with a single seed ranked the configurations differently: one seed was not enough to decide.

### Experiment 2: learning rate (with `eps_decay = 0.98`)

| `learning_rate` | 100-episode score per seed | Mean ± spread | Moving average reaches 200 |
|---|---|---|---|
| **1e-3** | 249.8, 263.3, 240.0 | **251.0 ± 9.6** | 315, 328, 459 |
| 5e-4 | 242.6, 237.4, 258.6 | 246.2 ± 9.0 | 206, 231, 359 |
| 1e-4 | 264.6, 259.0, 129.3 | 217.6 ± 62.5 | 288, 399, never |

* 1e-3 and 5e-4 are equivalent on the final level (5 points apart, less than the spread between seeds). 5e-4 learns about 100 episodes faster.
* 1e-4 is the least reliable: one seed stays at 129.3 and never reaches 200. A first single-seed run had suggested it was the most stable, and three seeds did not confirm it.
* 1e-3 is kept, so the final configuration is the baseline with `eps_decay = 0.98`.

## Comparison with PPO (from `ppo.ipynb`, section 10)

| Criterion | DQN | PPO |
|---|---|---|
| score over the same 100 episodes | 246.8 ± 53.4 | **262.2 ± 46.7** |
| three seeds | **251.0 ± 9.6** | 254.3 ± 13.9 |
| moving average reaches 200 (seeds 42, 43, 44) | **episodes 315, 328, 459** | episodes 650, 974, 1777 |
| training budget | **600 episodes (205,197 steps)** | 1,000,000 steps |
| training time, same laptop | **1.6 min** | 4.4 to 4.6 min |

DQN learns in fewer steps and less time, PPO has the higher typical score. The gap on the final score is about the size of the spread between seeds. These numbers are measured in `ppo.ipynb`, not in `dqn.ipynb`.

## Limits and next steps (notebook section 8)

* Three seeds per configuration: gaps of about 10 points are not significant.
* The experiments are sequential and test one parameter at a time: the interactions are not measured, and the other hyperparameters (`gamma`, network size, batch size, target update) were not tested.
* The final model is one training: the three seeds are the reference for what to expect from the configuration.
* Not tested: more episodes, a more reliable choice of the best checkpoint (more than 10 episodes), other target update frequencies.

## Reproducing

* All the hyperparameters are in `CONFIG`, the seed of the baseline training is `SEED = 42`, and the experiments use `SEEDS = [42, 43, 44]`. `run_experiment` sets Python, NumPy and PyTorch seeds before each training.
* `RETRAIN = True` in the notebook: the baseline agent and every configuration of the experiments are trained again (the whole notebook takes a few hours).
* The baseline training writes `dqn_lunarlander_baseline.pt` and its history. The trial `eps_decay=0.98`, seed 42, of experiment 1 writes the final model `dqn_lunarlander.pt`; with the same seed, it gives the same agent.
* Only `dqn_lunarlander.pt` is in the repository.
