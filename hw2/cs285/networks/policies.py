import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # fin
        obs = ptu.from_numpy(obs)
        
        if self.discrete:
            action = self.forward(obs)
            action = ptu.to_numpy(action)
            action = np.argmax(action)
        else:
            dist = self.forward(obs)
            action = dist.rsample()
        
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # fin? -> no! it should be.... 0 or 1...? or what? argmax??? hmmm Q. can it be non differentiable..?
            logits = self.logits_net(obs)
            action = F.softmax(logits)
            # action = torch.exp(action)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # fin?
            mu = self.mean_net(observation)
            std = torch.exp(self.logstd)
            action = distributions.Normal(mu, std)
            # action = normal.rsample()
        return action

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        
        # TODO: implement the policy gradient actor update.
        # Q. 저렇게 loss에 곱해주기만 하면 되나?, GT actions는 어떤 형태인지 모르는 어캄(이게 logit인지 뭐시긴지어떻게알아)
        # A. 일단 다른 코드에서 loss에 Reward 곱하긴함
        # advantages detach는 input에서 되어서 들어옴(np니까)
        self.optimizer.zero_grad()
        if self.discrete:
            logits = torch.log(self.forward(obs))
            logits_ac = logits[ torch.arange(actions.shape[0]) , actions.long()]
        else:
            dist = self.forward(obs)
            logits_ac = dist.log_prob(actions)
        
        loss = - logits_ac * advantages # actions_pred 
        loss = loss.mean()
        loss.backward() # mean!

        self.optimizer.step()
        
        self.optimizer.zero_grad()
        

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
