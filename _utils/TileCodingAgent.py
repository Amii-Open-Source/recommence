import numpy as np
from typing import Any, Dict, List
from PyExpUtils.utils.random import sample
from ReplayTables.interface import Timestep
from ReplayTables.ingress.LagBuffer import LagBuffer
from RlGlue import BaseAgent
from numba import njit
from PyExpUtils.utils.jit import try2jit
from PyFixedReps.TileCoder import TileCoder, TileCoderConfig



@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha):
    qsa = w[a][x].sum()

    qsp = w.T[xp].sum(axis=0)

    delta = r + gamma * qsp.dot(pi) - qsa

    w[a][x] = w[a][x] + alpha / len(x) * delta


@njit(cache=True)
def value(w, x):
    return w.T[x].sum(axis=0)


class SparseTileCoder(TileCoder):
    def __init__(self, params: TileCoderConfig, rng=None):
        params.scale_output = False
        super().__init__(params, rng=rng)

    def encode(self, s):
        return super().get_indices(s)


@try2jit
def argsmax(arr: np.ndarray):
    ties: List[int] = [
        0 for _ in range(0)
    ]  # <-- trick njit into knowing the type of this empty list
    top: float = arr[0]

    for i in range(len(arr)):
        if arr[i] > top:
            ties = [i]
            top = arr[i]

        elif arr[i] == top:
            ties.append(i)

    if len(ties) == 0:
        ties = list(range(len(arr)))

    return ties


@njit(cache=True)
def egreedy_probabilities(qs: np.ndarray, actions: int, epsilon: float):
    # compute the greedy policy
    max_acts = argsmax(qs)
    pi: np.ndarray = np.zeros(actions)
    for a in max_acts:
        pi[a] = 1.0 / len(max_acts)

    # compute a uniform random policy
    uniform: np.ndarray = np.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1.0 - epsilon) * pi + epsilon * uniform


class TileCodingAgent(BaseAgent):
    def __init__(self):
        ####################################################
        # TODO: bad practice, should later figure out what to do to avoid hard coding these
        self.n_step = 10000
        self.gamma = 0.99
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.actions = 2
        self.alpha = 0.01
        self.epsilon = 0.1
        self.params = {
            "representation": {
                "tiles": 8,
                "tilings": 8,
                "input_ranges": [
                    (-4.8, 4.8),
                    (-np.inf, np.inf),
                    (-0.42, 0.42),
                    (-np.inf, np.inf),
                ],
            }
        }
        self.observations = (4,)
        ##########################################################
        self.lag = LagBuffer(self.n_step)
        self.rep_params: Dict = self.params["representation"]
        self.rep = SparseTileCoder(
            TileCoderConfig(
                tiles=self.rep_params["tiles"],
                tilings=self.rep_params["tilings"],
                dims=self.observations[0],
                input_ranges=self.rep_params["input_ranges"],
            )
        )

        self.w = np.zeros((self.actions, self.rep.features()), dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
        else:
            pi = self.policy(xp)

        _update(self.w, x, a, xp, pi, r, gamma, self.alpha)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, observation: np.ndarray):
        self.lag.flush()

        x = self.rep.encode(observation)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.lag.add(
            Timestep(
                x=x,
                a=a,
                r=None,
                gamma=0,
                terminal=False,
            )
        )
        return a

    def step(
        self, reward: float, observation: np.ndarray | None, extra: Dict[str, Any]
    ):
        a = -1

        # sample next action
        xp = None
        if observation is not None:
            xp = self.rep.encode(observation)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        interaction = Timestep(
            x=xp,
            a=a,
            r=reward,
            gamma=self.gamma * gamma,
            terminal=False,
        )

        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        return a

    def end(self, reward: float, extra: Dict[str, Any]):
        interaction = Timestep(
            x=None,
            a=-1,
            r=reward,
            gamma=0,
            terminal=True,
        )
        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        self.lag.flush()

    def cleanup(self): ...
