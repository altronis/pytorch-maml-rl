import torch
from copy import deepcopy

from maml_rl.samplers import MultiTaskSampler
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import weighted_mean, detach_distribution
from maml_rl.utils.reinforcement_learning import reinforce_loss


class MAMLPPO(GradientBasedMetaLearner):
    """
    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.
    """
    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(MAMLPPO, self).__init__(policy, device=device)

        self.fast_lr = fast_lr
        self.first_order = first_order
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

        with torch.no_grad():
            self.old_policy = self.policy

    async def adapt(self, train_futures):
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            inner_loss = reinforce_loss(self.policy,
                                        await futures,
                                        params=params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=self.first_order)
        return params

    async def surrogate_loss(self, train_futures, valid_futures):
        params = await self.adapt(train_futures)

        with torch.set_grad_enabled(True):
            valid_episodes = await valid_futures

            old_pi = self.old_policy(valid_episodes.observations, params=params)
            pi = self.policy(valid_episodes.observations, params=params)

            old_pi = detach_distribution(old_pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            advantages = valid_episodes.advantages
            product = ratio * advantages

            eps = 0.2
            eps_product = torch.where(advantages >= 0, (1 + eps) * advantages, (1 - eps) * advantages)

            losses = -weighted_mean(torch.minimum(product, eps_product),
                                    lengths=valid_episodes.lengths)

        return losses.mean(), torch.zeros_like(losses.mean())

    def step(self,
             train_futures,
             valid_futures,
             **kwargs):

        num_tasks = len(train_futures[0])
        self.optimizer.zero_grad()

        # Compute the surrogate loss
        losses, _ = self._async_gather([
            self.surrogate_loss(train, valid)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        loss = sum(losses) / num_tasks
        loss.backward()

        with torch.no_grad():
            self.old_policy = deepcopy(self.policy)

        # Perform optimization step
        self.optimizer.step()
