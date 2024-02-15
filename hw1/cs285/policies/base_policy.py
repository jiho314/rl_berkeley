import abc
import numpy as np
from cs285.infrastructure import pytorch_util as ptu

class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        observation = ptu.from_numpy(observation.astype(np.float32))
        dist = self.forward(observation)
        return ptu.to_numpy(dist.rsample())
        # raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
