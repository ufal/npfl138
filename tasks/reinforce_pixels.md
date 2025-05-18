### Assignment: reinforce_pixels
#### Date: Deadline: ~~May 21~~ Jun 30, 22:00
#### Points: 2 points

This is a continuation of the `reinforce_baseline` assignment.

The supplied [npfl138/CartPolePixels-v1](https://github.com/ufal/npfl138/blob/master/labs/npfl138/envs/cart_pole_pixels.py)
environment generates a pixel representation of the `CartPole` environment
as an $80×80$ `np.uint8` image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones). You can also
play it interactively using `python3 -m npfl138.envs.cart_pole_pixels_interactive`.

To pass the assignment, you need to reach an average return of 400 in 100
evaluation episodes. During evaluation in ReCodEx, two different random seeds
will be employed, and you need to reach the required return on all of them. Time
limit for each test is 10 minutes.

You should probably train the model locally and submit the already pretrained
model to ReCodEx, but it is also possible to train an agent during ReCodEx
evaluation.

Start with the
[reinforce_pixels.py](https://github.com/ufal/npfl138/tree/master/labs/11/reinforce_pixels.py)
template, which parses several parameters and creates the correct environment.
