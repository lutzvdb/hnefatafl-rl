import numpy as np
import torch

# Constants defining the board

# [defender, attacker, king]
NO_PIECE = [False, False, False]
DEFENDER_PIECE = [True, False, False]
ATTACKER_PIECE = [False, True, False]
KING_PIECE = [False, False, True]

# _which_team_is_on: 2 = attacker, 1 = defender
TEAM_ATTACKER = 2
TEAM_DEFENDER = 1

""" Get the position of the king on the board """


def get_king_pos(board):
    king = list(map(lambda arr: list(map(np.all, arr)), board == KING_PIECE))
    coords = np.where(king)
    king_pos = [int(coords[0]), int(coords[1])]

    return king_pos


# Convert a simpler board representation to the internal board representation
# The simpler representation is a 2D one with 1,2,3 instead of [True, False, False] etc.
# This simpler representation is the same one used with app found on hnefatafl.app
def map_board_to_tensor(board):
    def map_cols(item):
        # [Defender, Attacker, King]
        ret = [False, False, False]
        if item > 0 & item < 4:
            ret[item - 1] = True

        return ret

    def map_row(row):
        row = list(map(map_cols, row))
        return row

    mapped = list(map(map_row, board))
    mapped = np.array(mapped)

    return mapped


""" 
Base game. This is the smallest board that I could find. Bigger boards 
(9x9, 11x11) are more frequently played but more computationally intensive, so I stuck
with the small one for sake of PoC.
 """
GAME_BRANDUBH = map_board_to_tensor(
    [
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [2, 2, 1, 3, 1, 2, 2],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
    ]
)

""" 
Fetlar Hnefatafl variant. 
"""
GAME_HNEFATAFL = map_board_to_tensor(
    [
        [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
        [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2],
        [2, 2, 0, 1, 1, 3, 1, 1, 0, 2, 2],
        [2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2],
        [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
    ]
)
