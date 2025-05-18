### Assignment: reinforce
#### Date: Deadline: ~~May 21~~ Jun 30, 22:00
#### Points: 2 points

Solve the continuous [CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
from the [Gymnasium library](https://gymnasium.farama.org/) using the REINFORCE
algorithm. The `gymnasium` environments have the following methods and
properties:
- `observation_space`: the description of environment observations; for
  continuous spaces, `observation_space.shape` contains their shape
- `action_space`: the description of environment actions; for discrete
  actions, `action_space.n` is the number of actions
- `reset() → new_state, info`: starts a new episode, returning the new
  state and additional environment-specific information
- `step(action) → new_state, reward, terminated, truncated, info`: performs the
  chosen action in the environment, returning the new state, obtained reward,
  boolean flags indicating a terminal state and episode truncation, and
  additional environment-specific information

We additionally extend the `gymnasium` environment by:
- `episode`: number of the current episode (zero-based)
- `reset(start_evaluation=False) → new_state, info`: if `start_evaluation` is
  `True`, an evaluation is started

Once you finish training (which you indicate by passing `start_evaluation=True`
to `reset`), your goal is to reach an average return of 475 during 100
evaluation episodes. Note that the environment prints your 100-episode
average return each 10 episodes even during training.

Start with the [reinforce.py](https://github.com/ufal/npfl138/tree/master/labs/11/reinforce.py)
template, which provides a simple network implementation in PyTorch.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
