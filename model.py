import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import nn

ENV_ID = "LunarLander-v3"
EXIT_SUCCESS = 0
EXIT_FAILURE = 84
HELP_FLAGS = ("-h", "--help")
MIN_ARGUMENTS = 2
MAX_ARGUMENTS = 3
DEFAULT_EPISODES = 1
SUPPORTED_ALGORITHMS = ("dqn", "ppo")
DQN_STATE_PREFIX = "network."
USAGE = """USAGE
    python model.py ALGO MODEL [EPISODES]

DESCRIPTION
    ALGO         algorithm of the saved model: dqn or ppo
    MODEL        path to the saved model file
    EPISODES     number of episodes to run (default: 1)"""


class ModelError(Exception):
    pass


class Agent(Protocol):
    def __call__(self, *, state: np.ndarray) -> int: ...


@dataclass(frozen=True)
class Arguments:
    algorithm: str
    model_path: Path
    n_episodes: int


def parse_episodes(*, text: str) -> int:
    try:
        n_episodes = int(text)
    except ValueError:
        raise ModelError(f"EPISODES must be a positive integer, got '{text}'") from None
    if n_episodes < 1:
        raise ModelError(f"EPISODES must be a positive integer, got '{text}'")
    return n_episodes


def parse_arguments(*, arguments: list[str]) -> Arguments:
    if not MIN_ARGUMENTS <= len(arguments) <= MAX_ARGUMENTS:
        raise ModelError(f"expected 2 or 3 arguments, got {len(arguments)}\n\n{USAGE}")

    algorithm, model = arguments[0], arguments[1]
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ModelError(f"unknown algorithm '{algorithm}', expected dqn or ppo")

    model_path = Path(model)
    if not model_path.is_file():
        raise ModelError(f"model file '{model}' not found")

    n_episodes = parse_episodes(text=arguments[2]) if len(arguments) == MAX_ARGUMENTS else DEFAULT_EPISODES
    return Arguments(algorithm=algorithm, model_path=model_path, n_episodes=n_episodes)


def make_greedy_agent(*, network: nn.Module) -> Agent:
    @torch.no_grad()
    def greedy_agent(*, state: np.ndarray) -> int:
        return int(network(torch.as_tensor(state, dtype=torch.float32)).argmax())

    return greedy_agent


def build_mlp(*, input_size: int, hidden_size: int, output_size: int, activation: type[nn.Module]) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, output_size),
    )


def read_checkpoint(*, path: Path) -> object:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        raise ModelError(f"cannot read '{path}' as a PyTorch model file") from None


def load_ppo_agent(*, path: Path, input_size: int, output_size: int) -> Agent:
    checkpoint = read_checkpoint(path=path)
    if not isinstance(checkpoint, dict) or checkpoint.get("algorithm") != "ppo":
        raise ModelError(f"'{path}' is not a ppo model")

    try:
        actor = build_mlp(
            input_size=input_size,
            hidden_size=checkpoint["config"]["hidden_size"],
            output_size=output_size,
            activation=nn.Tanh,
        )
        actor.load_state_dict(checkpoint["actor"])
    except (KeyError, TypeError, RuntimeError) as error:
        raise ModelError(f"'{path}' does not contain a valid ppo actor: {error}") from None
    actor.eval()
    return make_greedy_agent(network=actor)


def load_dqn_agent(*, path: Path, input_size: int, output_size: int) -> Agent:
    state_dict = read_checkpoint(path=path)
    if not isinstance(state_dict, dict) or not all(
        isinstance(name, str) and name.startswith(DQN_STATE_PREFIX) for name in state_dict
    ):
        raise ModelError(f"'{path}' is not a dqn model")

    weights = {name.removeprefix(DQN_STATE_PREFIX): tensor for name, tensor in state_dict.items()}
    try:
        network = build_mlp(
            input_size=input_size,
            hidden_size=weights["0.weight"].shape[0],
            output_size=output_size,
            activation=nn.ReLU,
        )
        network.load_state_dict(weights)
    except (KeyError, AttributeError, IndexError, RuntimeError) as error:
        raise ModelError(f"'{path}' does not contain a valid dqn network: {error}") from None
    network.eval()
    return make_greedy_agent(network=network)


AGENT_LOADERS: dict[str, Callable[..., Agent]] = {
    "dqn": load_dqn_agent,
    "ppo": load_ppo_agent,
}


def play_episodes(*, env: gym.Env, agent: Agent, n_episodes: int) -> None:
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent(state=state))
            total_reward += float(reward)
            done = terminated or truncated
        print(f"Episode {episode}: {total_reward:.1f}")


def main(*, arguments: list[str]) -> int:
    if len(arguments) == 1 and arguments[0] in HELP_FLAGS:
        print(USAGE)
        return EXIT_SUCCESS

    try:
        parsed = parse_arguments(arguments=arguments)
        env = gym.make(ENV_ID, render_mode="human")
        try:
            agent = AGENT_LOADERS[parsed.algorithm](
                path=parsed.model_path,
                input_size=env.observation_space.shape[0],
                output_size=env.action_space.n,
            )
            play_episodes(env=env, agent=agent, n_episodes=parsed.n_episodes)
        finally:
            env.close()
    except ModelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main(arguments=sys.argv[1:]))
