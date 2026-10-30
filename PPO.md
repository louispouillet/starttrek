# PPO on LunarLander-v3

A summary of the PPO agent of `ppo.ipynb`, to read in a few minutes. Every number comes from an output of the notebook, and each part points to the section that proves it.

## Result

| Measure | Value |
|---|---|
| score over 100 consecutive episodes, without exploration | **262.2 ± 46.7** (solved: threshold 200) |
| episodes above 200 | 92 / 100 |
| end reasons | 92 at rest, 6 truncated, 2 crashes |
| three seeds (42, 43, 44) | 254.3 ± 13.9, all above 200 |
| training | 1,000,000 steps, about 4.5 minutes on a laptop CPU |
| random agent, for reference | −177.2 ± 94.6 |

## How PPO learns

```
┌─► 1. PLAY      the actor plays 2048 steps, sampling its actions; every step is stored
│   2. JUDGE     the critic gives one advantage per step: better or worse than expected?
│   3. LEARN     10 passes over the steps, in minibatches of 64: 320 adjustments
└── 4. DISCARD   the steps are thrown away, and the updated actor plays again
```

PPO is **on policy**: the stored steps describe the policy that played them, so they are discarded after the update (no replay buffer, unlike DQN). The budget of 1,000,000 steps gives 488 of these cycles.

| Element | Meaning |
|---|---|
| **actor** | network that gives one probability per action: the policy |
| **critic** | network that estimates $V(s)$, the total reward expected from a state |
| **advantage** | how much better an action did than what the critic expected, computed with GAE |
| **ratio** | new probability of a stored action divided by its probability when it was played |
| **clipping** | the ratio is limited to $[0.8, 1.2]$, so that one update cannot change the policy too much |
| **entropy** | how unsure the actor is (1.386 for 25% per action, 0 for a certain choice) |

The actor loss combines the **clipped policy loss** and the **entropy bonus**, $L_{actor} = L_{clip} - 0.01 \times H$. The critic loss is the **MSE** between $V(s)$ and the returns. Each network has its own Adam optimizer, which is equivalent to one combined loss since the networks share no weight.

## Exploration

During training, the action is **sampled** from the probabilities of the actor: exploration is built in, without epsilon greedy. It fades by itself as the actor becomes sure of itself (the entropy goes from 1.39 to about 0.45). During evaluation and in `model.py`, the agent takes the most probable action (`argmax`), as the subject requires.

## `terminated` and `truncated`

| Flag | Meaning | Treatment |
|---|---|---|
| `terminated` | real end: crash, out of the viewport, at rest | no future: the advantage stops here |
| `truncated` | limit of 1000 steps: the lander could have kept going | the value of the final state is added to the last reward, $r \leftarrow r + \gamma V(s_{final})$ |

Without this correction, the critic would learn that nothing can be earned from a hovering state.

## Architecture (notebook section 4)

```
                 ┌─ Linear(8, 64) ─ Tanh ─ Linear(64, 64) ─ Tanh ─ Linear(64, 4) ─► 4 logits   actor  (4996 parameters)
observation (8) ─┤
                 └─ Linear(8, 64) ─ Tanh ─ Linear(64, 64) ─ Tanh ─ Linear(64, 1) ─► V(s)       critic (4801 parameters)
```

* **Two separate networks**: the two losses do not compete on shared weights, so no coefficient has to balance them. It is the default of Stable-Baselines3 and CleanRL for vector observations.
* **Tanh**: bounded activations and smoother policy updates, the default for PPO.
* **Orthogonal initialization**: gain $\sqrt{2}$ for the hidden layers, 0.01 for the actor output (initial policy at 25% per action: the agent starts with no preference), 1 for the critic output.

## Hyperparameters

| Field | Value | Reason |
|---|---|---|
| `hidden_size` | 64 | standard size for PPO on small vector observations |
| `n_steps` | 2048 | steps per rollout: better advantages than short rollouts, still frequent updates |
| `gamma` | 0.99 | the agent looks about 100 steps (2 seconds) ahead, enough to value a landing |
| `gae_lambda` | 0.95 | GAE trade off between the bias of the critic and the noise of real returns |
| `learning_rate` | 3e-4 | size of each adjustment, default of Stable-Baselines3 |
| `n_epochs` | 10 | passes over each rollout: a rollout is expensive, and clipping makes its reuse safe |
| `batch_size` | 64 | 32 minibatches per pass, 320 adjustments per rollout |
| `clip_range` | 0.2 | ratio clipped to $[0.8, 1.2]$, value of the PPO paper |
| `entropy_coef` | 0.01 | weight of the entropy bonus, value of RL Zoo for LunarLander |
| `max_grad_norm` | 0.5 | longer gradients are scaled down, against minibatches full of crashes |
| `total_timesteps` | 1,000,000 | training budget, budget of RL Zoo for PPO on LunarLander |
| `eval_interval` | 10 | one evaluation every 20,480 steps, about 48 in total |
| `n_eval_episodes` | 10 | episodes per evaluation during training; the final evaluation uses 100 |

## Training (notebook section 7)

* The moving average over 100 episodes reaches 200 at **step 198,192**, after about one minute.
* Split into five phases of equal number of episodes, crashes fall from 51% of the episodes in the first phase to 4% in the last, and the lander comes to rest in more than 80% of them from the second phase on.
* The episode length shows three stages: short episodes ending in a crash, then a peak near 570 steps when the agent hovers without landing, then about 300 steps once it lands.
* The best version is kept with evaluations without exploration on 10 fixed episodes (best: 281.6 at step 655,360), then measured again on 100 new episodes.

## Several seeds (notebook section 8)

| Seed | Moving average reaches 200 | 100 episodes | At rest | Crash | Truncated |
|---|---|---|---|---|---|
| 42 | step 198,192 | 262.2 ± 46.7 | 92 | 2 | 6 |
| 43 | step 327,602 | 265.9 ± 53.8 | 93 | 4 | 3 |
| 44 | step 735,499 | 234.8 ± 78.5 | 86 | 10 | 4 |

The three seeds solve the environment, but the learning speed depends a lot on the seed: seed 44 stays on a plateau around 100 while it learns to hover before it learns to land.

## Experiment: without the entropy bonus (notebook section 9)

Only `entropy_coef` changes (0.01 to 0), on the same seeds, budget and evaluation.

| | With the bonus | Without the bonus |
|---|---|---|
| mean score over the seeds | 254.3 ± 13.9 | 248.7 ± 19.0 |
| moving average reaches 200, mean over the seeds | step 420,431 | step 359,142 |
| final entropy, mean over the seeds | 0.51 | 0.41 |

* Both versions solve the environment on the three seeds; the difference is smaller than the spread between seeds.
* The bonus does keep the policy less certain, but only after about 300,000 steps: a weight of 0.01 is small next to the policy loss.
* Hovering happens with and without the bonus, so it is not caused by a lack of exploration.
* Lesson: on LunarLander, PPO explores mainly by sampling its actions, and one seed per configuration is not enough to compare two configurations.

## Comparison with DQN (notebook section 10)

| Criterion | PPO | DQN |
|---|---|---|
| score over the same 100 episodes | **262.2 ± 46.7** | 246.8 ± 53.4 |
| episodes above 200 | 92 | **94** |
| three seeds | 254.3 ± 13.9 | 251.0 ± 9.6 |
| moving average reaches 200 (seeds 42, 43, 44) | episodes 650, 974, 1777 | **episodes 315, 328, 459** |
| training budget | 1,000,000 steps | 600 episodes (205,197 steps) |
| training time, same laptop | 4.4 to 4.6 min | **1.6 min** |

* **Final score and stability**: both agents solve the environment at the same level. PPO has the higher mean and median and lands faster (270 steps against 313); DQN has slightly more episodes above 200 and no crash. The gap is about the size of the spread between seeds.
* **Steps needed to learn**: DQN is more sample efficient, because it learns at every step and reuses its experiences from a replay buffer, while PPO discards each rollout after 10 epochs.
* **Training time**: DQN is faster on the same laptop, since it needs a budget about 5 times smaller. Its time was measured by training the final DQN configuration once more with the code of `dqn.ipynb`, because the DQN notebook does not record it.
* **Tuning**: the decay of epsilon was the main lever of DQN (+23 points over the seeds), while the entropy bonus barely mattered for PPO, which explores by sampling its own policy.

## Limits and next steps (notebook section 11)

* Hovering is the main failure mode (for DQN too), and the learning speed depends a lot on the seed.
* Not tested: more seeds, a decreasing learning rate, vectorized environments.

## Reproducing

* All the hyperparameters are in `PPOConfig`, and the seed is fixed with `set_seed` and `env.reset(seed=...)`.
* `RETRAIN = False` (default) reloads a saved training made with the same configuration; otherwise, the agent is trained again and gets the same result with the same seed.
* The notebook writes one model file and one training log per training; only `ppo_lunarlander.pt` (seed 42) is in the repository.
