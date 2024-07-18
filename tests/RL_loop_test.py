from recommence.Checkpoint import Checkpoint
from _utils.TileCodingAgent import TileCodingAgent
from RlGlue import RlGlue, BaseEnvironment
import gymnasium
from typing import Any, Dict, Tuple

class stepCollector:
    def __init__(self):
        self.total_steps = 0

    def add(self, steps):
        self.total_steps += steps

    def get(self):
        return self.total_steps

    def reset(self):
        self.total_steps = 0


class rewardCollector:
    def __init__(self):
        self.total_reward = 0

    def add(self, r):
        self.total_reward += r

    def get(self):
        return self.total_reward

    def reset(self):
        self.total_reward = 0


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


def test_RL_integration(tmp_path):
    chk = Checkpoint(save_path=str(tmp_path))

    agent = chk.register("agent", lambda: TileCodingAgent())
    env = chk.register("env", lambda: CartpoleEnvironment(seed=42))
    glue = chk.register("glue", lambda: RlGlue(agent, env))

    chk.register("reward", lambda: rewardCollector())
    chk.register("steps", lambda: stepCollector())

    save_checkpoint_every = 1000
    restart_loop_at = 5000
    max_training_loop_steps = 10000

    if glue.total_steps == 0:
        glue.start()

    step = 0
    while step < max_training_loop_steps:
        # print("at step:", step)

        interaction = glue.step()
        # print("info:", interaction.r, interaction.o, interaction.a, interaction.t)

        if step % save_checkpoint_every == 0:
            print("last saved total reward is:", chk["reward"].get())
            chk.save()

        if step == restart_loop_at:
            print("reset at step:", step)
            print("total reward before resetting is:", chk["reward"].get())
            step = chk["steps"].get()
            print("have information ready from step:", step)

            print("reloading agent and env")
            agent = chk["agent"]
            env = chk["env"]
            glue = chk["glue"]

        if interaction.t:
            agent.cleanup()
            glue.start()

        step += 1
        chk["reward"].add(interaction.r)
        chk["steps"].add(1)

    chk.save()


