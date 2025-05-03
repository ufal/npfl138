# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pygame

from . import cart_pole_pixels


# Allow running the environment and controlling it with arrows
if __name__ == "__main__":
    env = cart_pole_pixels.CartPolePixels(render_mode="human")
    env.metadata["render_fps"] = 10

    quit = False
    while not quit:
        env.reset()
        steps, action, restart = 0, 0, False
        while True:
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 0
                    if event.key == pygame.K_RIGHT:
                        action = 1
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                if event.type == pygame.QUIT:
                    quit = True

            # Perform the step
            _, _, terminated, truncated, _ = env.step(action)

            steps += 1
            if terminated or truncated or restart or quit:
                break
        print("Episode ended after {} steps".format(steps))

    env.close()
