from recommence.Checkpoint import Checkpoint
from tests.TileCodingAgent import TileCodingAgent
from RlGlue import RlGlue, BaseEnvironment
import gymnasium
from typing import Any, Dict, Tuple




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
        self.env = gymnasium.make("CartPole-v1")
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

    agent = chk.register('agent', lambda: TileCodingAgent())
    env = chk.register('env', lambda: CartpoleEnvironment(seed=10))
    glue = chk.register('glue', lambda: RlGlue(agent, env))
    total_rewards = chk.register('reward', lambda: rewardCollector())
    print("here")
    glue.total_steps = 0

    save_interval = 1000  # Number of steps between saving checkpoints
    stop_step = 5000  # Step at which to stop the learning loop
    load_checkpoint = False  # Whether to load from the checkpoint or start from scratch

    if load_checkpoint:
        chk["reward"].add(total_rewards)
        glue.total_steps += save_interval  # Skip the steps already performed

    if glue.total_steps == 0:
        glue.start()

    for step in range(glue.total_steps, 10000):
        print("here")

        interaction = glue.step()
        print(interaction.r, interaction.o, interaction.a, interaction.t)

        if step % save_interval == 0:
            chk.save()  # Save the checkpoint

        if step == stop_step:
            break

        if interaction.t:
            agent.cleanup()
            glue.start()

    chk.save()  # Save the final checkpoint
    print(chk['reward'].get())


test_RL_integration("tmp_path")


