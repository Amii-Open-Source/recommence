import jax.numpy as jnp
from recommence.Checkpoint import Checkpoint, CheckpointConfig


class Agent:
    def __init__(self):
        self._w = jnp.ones(25)


def test_jax(tmp_path):
    config = CheckpointConfig(
        save_path=str(tmp_path),
    )
    chk = Checkpoint(config)

    agent = chk.register("agent", Agent)
    agent._w *= 2
    chk.save()
    del agent

    chk = Checkpoint(config)

    assert jnp.all(chk["agent"]._w == (jnp.ones(25) * 2))
