from recommence.Checkpoint import Checkpoint
from tests._utils.TileCodingAgent import TileCodingAgent
from RlGlue import RlGlue, BaseEnvironment
import gymnasium
from typing import Any, Dict, Tuple
import numpy as np

class FakeCheckpoint:
  def register(self, key: str, builder: ...):
    return builder()
  def save(self):
    pass
  def __getitem__(self, key):
    return ReturnCollector() # return a fake object


class ReturnCollector:
    def __init__(self):
        self._return_sequence = []
    def add(self, r):
        self._return_sequence.append(r)

    def get(self):
        return np.array(self._return_sequence)


class CartpoleEnvironment(BaseEnvironment):
    def __init__(self, seed: int):
        self.env = gymnasium.make("CartPole-v1", max_episode_steps=1000000)
        self.seed = seed

    def start(self):
        self.seed += 1
        s, info = self.env.reset(seed=self.seed)
        return s

    def step(self, action) -> Tuple[float, Any, bool, Dict[str, Any]]:
        sp, reward, terminal, truncate, info = self.env.step(action)
        return float(reward), sp, terminal, {}


def run_rl_system(seed: int, should_chk: bool, tmp_path) -> Tuple[np.ndarray, ReturnCollector]:
    chk = Checkpoint(str(tmp_path)) if should_chk else FakeCheckpoint()

    agent = chk.register('agent', lambda: TileCodingAgent())
    env = chk.register("env", lambda: CartpoleEnvironment(seed=42))
    glue = chk.register("glue", lambda: RlGlue(agent, env))
    return_collector = chk.register("returns", lambda: ReturnCollector())

    if glue.total_steps == 0:
        glue.start()

    max_steps = 10000
    test_step = 5000
    steps = 0

    while steps < max_steps:
        interaction = glue.step()

        if interaction.t:
            agent.cleanup()
            glue.start()

        if steps == test_step and should_chk:
            chk.save()
            del chk

            chk = Checkpoint(tmp_path)
            agent = chk['agent']
            env = chk['env']
            glue = chk['glue']
            return_collector = chk['returns']

        if should_chk:
                chk["returns"].add(interaction.r)
        else:
                return_collector.add(interaction.r)
        steps += 1

    return agent.w, return_collector

def test_RL_integration(tmp_path):
  no_chk_result = run_rl_system(seed=42, should_chk=False, tmp_path=tmp_path)
  chk_result = run_rl_system(seed=42, should_chk=True, tmp_path=tmp_path)

  assert np.all(no_chk_result[0] == chk_result[0])
  assert np.all(no_chk_result[1].get() == chk_result[1].get())



