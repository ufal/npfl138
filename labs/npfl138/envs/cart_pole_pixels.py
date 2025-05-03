# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import gymnasium.envs.classic_control
import numpy as np

import pygame
import pygame.gfxdraw


class CartPolePixels(gymnasium.envs.classic_control.CartPoleEnv):
    H, W = 80, 80
    SCALE = 8

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self._images = 3
        self._image = np.zeros([self.H, self.W, self._images], dtype=np.uint8)
        self._image_surface = pygame.Surface((self.H, self.W))
        self._screen = None

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=[self.H, self.W, self._images], dtype=np.uint8)

    def reset(self, seed=None, options=None):
        render_mode = self.render_mode
        self.render_mode = None
        state, info = super().reset(seed=seed, options=options)
        self.render_mode = render_mode

        for step in range(self._images):
            observation = self._draw(state)

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        render_mode = self.render_mode
        self.render_mode = None
        state, reward, termination, truncation, info = super().step(action)
        self.render_mode = render_mode

        observation = self._draw(state)

        if self.render_mode == "human":
            self.render()
        return observation, reward, termination, truncation, info

    def render(self):
        assert self.render_mode in self.metadata["render_modes"]

        if self.render_mode == "rgb_array":
            return np.copy(self._image)

        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.W * self.SCALE, self.H * self.SCALE))
            self._screen_surface = pygame.Surface((self.W * self.SCALE, self.H * self.SCALE))
            self._clock = pygame.time.Clock()

        pygame.pixelcopy.array_to_surface(
            self._screen_surface, self._image.repeat(8, axis=0).repeat(8, axis=1).transpose([1, 0, 2]))

        pygame.event.pump()
        self._screen.blit(self._screen_surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        pygame.event.pump()

    def close(self):
        if self._screen is not None:
            self._screen = None
            pygame.display.quit()
            pygame.quit()

    def _draw(self, state):
        for i in range(self._images - 1):
            self._image[:, :, i] = self._image[:, :, i + 1]

        cart = int(40 + state[0] / 3 * 40)
        cart_poly = [(70, cart - 10), (80, cart - 10), (80, cart + 10), (70, cart + 10)]

        pole_x = int(40 + (state[0] + np.sin(state[2]) * 4.2) / 3 * 40)
        pole_y = int(70 - np.cos(state[2]) * 5.2 / 3 * 40)
        pole_poly = [(pole_y, pole_x - 2), (70, cart - 2), (70, cart + 2), (pole_y, pole_x + 2)]

        self._image_surface.fill([0, 0, 0])
        pygame.gfxdraw.polygon(self._image_surface, pole_poly, [255, 255, 255])
        pygame.gfxdraw.filled_polygon(self._image_surface, pole_poly, [255, 255, 255])
        pygame.gfxdraw.filled_polygon(self._image_surface, cart_poly, [128, 128, 128])
        pygame.pixelcopy.surface_to_array(self._image[:, :, -1], self._image_surface, kind="R")

        return np.copy(self._image)


#################################
# Environment for NPFL138 class #
#################################

gym.envs.register(
    id="npfl138/CartPolePixels-v1",
    entry_point=CartPolePixels,
    max_episode_steps=500,
    reward_threshold=475,
)
