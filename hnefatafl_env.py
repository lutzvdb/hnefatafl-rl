import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium import spaces
import hnefatafl_utils as hu
import ruleset as rules

""" These rewards are a bit random, but should be good enough for now. """
REWARD_WIN = 50
REWARD_BEATING = 5
REWARD_ILLEGALMOVE = -10
REWARD_BASE = -1


class HnefataflEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, verbose=False, initial_board=None):
        self.window_size = 512  # The size of the PyGame window
        if initial_board is None:
            self.initial_board = hu.GAME_BRANDUBH
        else:
            self.initial_board = initial_board

        self._board_coords = [
            [x, y]
            for x in range(self.initial_board.shape[0])
            for y in range(self.initial_board.shape[0])
        ]
        self._verbose = verbose
        size = self.initial_board.shape[0]
        self._size = size
        self._terminated = False

        # 3d Tensor: x (0...size-1), y (0...size-1), piece (0..2)
        # piece: False means piece not present, True means piece present
        # [defender, attacker, king]
        # --> [0, 0, 1] means king is at that position
        # --> [1, 0, 0] means defender is at that position
        # --> [0, 0, 0] means no piece is at that position
        # --> Sum of this vector must always be <= 1
        self.observation_space = spaces.Box(0, 1, shape=(size, size, 3), dtype=bool)

        # All position references are 0-indexed
        # (0, 0) is the upper left corner
        # (size-1, size-1) is the lower right corner
        # (size-1, 0) is the lower left corner
        # [from_x, from_y, to_x, to_y]
        self.action_space = spaces.MultiDiscrete(
            [size - 1, size - 1, size - 1, size - 1]
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_info(self):
        return {"nstep": self._nstep}

    def _get_obs(self):
        return self._board

    def reset(self, seed=None, options=None):
        # Choose the agent's location uniformly at random
        self._board = self.initial_board.copy()
        # _which_team_is_on: 2 = attacker, 1 = defender
        self._which_team_is_on = hu.TEAM_ATTACKER
        self._terminated = False
        self._nstep = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = REWARD_BASE
        terminated = self._terminated
        truncated = False

        if terminated:
            # Stepping not possible after termination
            # Should not happen since in vectorized environments,
            # Environment is auto-reset after termination
            self._get_obs(), 0, terminated, truncated, self._get_info()

        self._nstep += 1

        from_piece, to_piece = action
        from_x = int(from_piece[0])
        from_y = int(from_piece[1])
        to_x = int(to_piece[0])
        to_y = int(to_piece[1])

        if (from_x == -1) | (to_x == -1) | (from_y == -1) | (to_y == -1):
            # Non-move; probably, the opponent has already won
            self._terminated = True
            return self._get_obs(), 0, self._terminated, truncated, self._get_info()

        moveIsOk, reason = rules.sanity_check_move(
            self._board,
            (from_x, from_y, to_x, to_y),
            self._which_team_is_on,
            self._verbose,
        )

        if moveIsOk == False:
            # print(reason)
            # print(action)
            # print(self._which_team_is_on)
            # print(self._board)
            reward = REWARD_ILLEGALMOVE
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # do the actual moving
        new_board = self._board.copy()
        picked_piece = self._board[from_x, from_y]
        new_board[from_x, from_y] = hu.NO_PIECE
        new_board[to_x, to_y] = picked_piece

        self._board = new_board

        # Check for royal win through king in corner
        win = rules.check_king_in_corner(self._board)
        if win == True:
            reward = REWARD_WIN
            terminated = True
            self._terminated = terminated
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Check for opposite win (king beaten)
        if self._which_team_is_on == hu.TEAM_ATTACKER:
            board, kingBeaten = rules.checkKingBeating(self._board)
            if kingBeaten == True:
                reward = REWARD_WIN
                terminated = True
                self._terminated = terminated
                return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Check for beating and remove beaten pieces
        board, beatingOccured = rules.check_beating(
            self._board, self._which_team_is_on, to_x, to_y
        )
        # Board was changed if beating occurred!
        self._board = board
        if beatingOccured == True:
            reward += REWARD_BEATING

        if rules.check_all_attackers_killed(self._board):
            # Defenders have won since all attackers have been killed
            reward = REWARD_WIN
            terminated = True
            self._terminated = terminated
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Switch teams for next move
        if self._which_team_is_on == hu.TEAM_ATTACKER:
            self._which_team_is_on = hu.TEAM_DEFENDER
        else:
            self._which_team_is_on = hu.TEAM_ATTACKER

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if (self.render_mode == "rgb_array") | (self.render_mode == "human"):
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self._size
        )  # The size of a single grid square in pixels

        # Loop through board and draw pieces
        for x in range(self._size):
            for y in range(self._size):
                piece = self._board[x, y]

                color = (240, 240, 240)  # no piece
                # [Defender, Attacker, King]
                if piece[0] == True:
                    color = (50, 200, 50)  # Defender
                elif piece[1] == True:
                    color = (200, 50, 50)  # Attacker
                elif piece[2] == True:
                    color = (252, 223, 3)  # King

                pygame.draw.circle(
                    canvas,
                    color,
                    # for pygame, x means horizontal and y vertical
                    # for us, x is the first index (indicating row, meaning vertical)
                    # y is the second index (indicating col, meaning horizontal)
                    # therefore, for display, we switch x and y
                    [(y + 0.5) * pix_square_size, (x + 0.5) * pix_square_size],
                    pix_square_size / 3,
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
