import numpy as np
from recommence.Checkpoint import Checkpoint

class FakeAgent:
    def __init__(self):
        self.weights = np.zeros((32, 4), dtype=np.float64)
        self.steps = 1

def test_checkpoint1(tmp_path):
    chk = Checkpoint(save_path=str(tmp_path))

    chk['agent'] = lambda: FakeAgent()
    original_agent = chk['agent']
    original_agent.weights[0] = 1

    assert isinstance(original_agent, FakeAgent)

    chk.save()
    del chk

    chk = Checkpoint(save_path=str(tmp_path))

    loaded_agent = chk['agent']

    assert id(original_agent) != id(loaded_agent)
    assert np.all(original_agent.weights == loaded_agent.weights)
    assert original_agent.steps == loaded_agent.steps

    chk.remove()


if __name__ == '__main__':
    test_checkpoint1(tmp_path='tmp_path')