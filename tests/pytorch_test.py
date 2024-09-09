import torch
import numpy as np
from recommence.Checkpoint import Checkpoint, CheckpointConfig


class Agent:
    def __init__(self):
        self._w = torch.tensor(np.ones(25))


def test_pytorch(tmp_path):
    config = CheckpointConfig(
        save_path=str(tmp_path),
    )
    chk = Checkpoint(config)

    agent = chk.register("agent", Agent)
    agent._w *= 2
    chk.save()
    del agent

    chk = Checkpoint(config)

    assert torch.all(chk["agent"]._w == (torch.tensor(np.ones(25)) * 2))
