import hnefatafl_utils as hu
import numpy as np

""" Checks if a beating occured with the last move """


def check_beating(board, which_team_is_on, to_x, to_y):
    beatingOccured = False

    board, beatingCheck = checkSimpleBeating(board, which_team_is_on, to_x, to_y)
    if beatingCheck:
        beatingOccured = True
    board, beatingOccured = checkBeatingWithCorner(board, which_team_is_on, to_x, to_y)
    if beatingCheck:
        beatingOccured = True
    board, beatingOccured = checkBeatingWithEmptyThrone(
        board, which_team_is_on, to_x, to_y
    )
    if beatingCheck:
        beatingOccured = True

    return board, beatingOccured


""" Check if the royal team won """


def check_king_in_corner(board):
    m = len(board) - 1  # outmost position
    # Game is won if the king is in any of the corners
    if all(board[0, 0] == hu.KING_PIECE):
        return True
    if all(board[0, m] == hu.KING_PIECE):
        return True
    if all(board[m, 0] == hu.KING_PIECE):
        return True
    if all(board[m, m] == hu.KING_PIECE):
        return True
    return False


""" See if all attackers were killed """


def check_all_attackers_killed(board):
    anyAttackersLeft = np.any(np.all(np.equal(board, hu.ATTACKER_PIECE), axis=-1))
    if anyAttackersLeft == False:
        return True

    return False


""" Sanity check: Were all rules followed in a move? """


def sanity_check_move(board, action, cur_team, verbose):
    from_x, from_y, to_x, to_y = action
    picked_piece = board[from_x, from_y]
    target_piece = board[to_x, to_y]
    reason = ""

    # Check if we picked the right piece and are moving
    # to an empty spot
    if all(picked_piece == hu.NO_PIECE):
        reason = "Move sanity check failed: Attempted to pick nonexisting piece"
        if verbose:
            print(reason)
        return False, reason
    if cur_team == hu.TEAM_ATTACKER & any(picked_piece != hu.ATTACKER_PIECE):
        reason = "Move sanity check failed: Picked wrong piece"
        if verbose:
            print(reason)
        return False, reason
    if cur_team == hu.TEAM_DEFENDER & all(picked_piece == hu.ATTACKER_PIECE):
        reason = "Move sanity check failed: Picked wrong piece"
        if verbose:
            print(reason)
        return False, reason
    if any(target_piece != hu.NO_PIECE):
        reason = "Move sanity check failed: Must move to an empty spot"
        if verbose:
            print(reason)
        return False, reason

    # Disallow non-orthogonal movement
    # --> only x or y can change, not both
    if (from_x != to_x) & (from_y != to_y):
        reason = "Move sanity check failed: Diagonal movement"
        if verbose:
            print(reason)
        return False, reason

    # The king can only move 3
    distance = abs(from_x - to_x) + abs(from_y - to_y)
    if all(picked_piece == hu.KING_PIECE) & distance > 3:
        reason = "Move sanity check failed: King must move max. distance of 3"
        if verbose:
            print(reason)
        return False, reason

    # Check if anything is in the way
    full_trajectory = board[from_x : (to_x + 1), from_y : (to_y + 1)].squeeze()
    if len(full_trajectory) > 2:
        # We're not just moving next door
        flyover_pieces = full_trajectory[1:-1]
        illegal_flyovers = np.sum(flyover_pieces != hu.NO_PIECE)
        if illegal_flyovers > 0:
            reason = "Move sanity check failed: Illegal flyover"
            if verbose:
                print(reason)
            return False, reason

    """ if verbose:
        print("Sanity check passed") """

    return True, reason


""" Check if the king is beaten (attackers won) """


# if the king is surrounded by 4 sides, he's done
def checkKingBeating(board):
    king_pos = hu.get_king_pos(board)
    board_size = board.shape[0]
    thronePos = [int((board_size - 1) / 2), int((board_size - 1) / 2)]
    beatingOccured = False

    # if on the edge, king is not touched by 4-side-rule
    if (
        (king_pos[0] == 0)
        | (king_pos[0] == board_size - 1)
        | (king_pos[1] == 0)
        | (king_pos[1] == board_size - 1)
    ):
        return board, beatingOccured

    # Temporarily replace throne position with a red piece fur
    # faster checking
    tmpthronePosVal = board[thronePos[0], thronePos[1]].copy()
    board[thronePos[0], thronePos[1]] = hu.ATTACKER_PIECE

    if (
        (all(board[king_pos[0] - 1, king_pos[1]] == hu.ATTACKER_PIECE))
        & (all(board[king_pos[0] + 1, king_pos[1]] == hu.ATTACKER_PIECE))
        & (all(board[king_pos[0], king_pos[1] - 1] == hu.ATTACKER_PIECE))
        & (all(board[king_pos[0], king_pos[1] + 1] == hu.ATTACKER_PIECE))
    ):
        beatingOccured = True  # King is surrounded on all sides... He has lost

    # revert throne position
    board[thronePos[0], thronePos[1]] = tmpthronePosVal
    return board, beatingOccured


# advanced case: our move places an enemy stone between empty throne and us
# check in all 4 directions if step1 is enemy and step2 is empty throne
# if so, remove enemy stone
def checkBeatingWithEmptyThrone(board, which_team_is_on, to_x, to_y):
    enemy = hu.ATTACKER_PIECE
    beatingOccured = False

    if which_team_is_on == hu.TEAM_ATTACKER:
        enemy = hu.DEFENDER_PIECE

    # up
    if to_x > 1:
        if (
            all(board[to_x - 1, to_y] == enemy)
            & all(board[to_x - 2, to_y] == hu.NO_PIECE)
            & (to_x - 2 == (len(board) - 1) / 2)
            & (to_y == (len(board) - 1) / 2)
        ):
            beatingOccured = True
            board[to_x - 1, to_y] = hu.NO_PIECE

    # down
    if to_x < len(board) - 2:
        if (
            all(board[to_x + 1, to_y] == enemy)
            & all(board[to_x + 2, to_y] == hu.NO_PIECE)
            & (to_x + 2 == (len(board) - 1) / 2)
            & (to_y == (len(board) - 1) / 2)
        ):
            beatingOccured = True
            board[to_x + 1, to_y] = hu.NO_PIECE

    # left
    if to_y > 1:
        if (
            all(board[to_x, to_y - 1] == enemy)
            & all(board[to_x, to_y - 2] == hu.NO_PIECE)
            & (to_y - 2 == (len(board) - 1) / 2)
            & (to_x == (len(board) - 1) / 2)
        ):
            beatingOccured = True
            board[to_x, to_y - 1] = hu.NO_PIECE

    # right
    if to_y < len(board) - 2:
        if (
            all(board[to_x, to_y + 1] == enemy)
            & all(board[to_x, to_y + 2] == hu.NO_PIECE)
            & (to_y + 2 == (len(board) - 1) / 2)
            & (to_x == (len(board) - 1) / 2)
        ):
            beatingOccured = True
            board[to_x, to_y + 1] = hu.NO_PIECE

    return board, beatingOccured


# advanced case: our move places an enemy stone between cornr and us
# check in all 4 directions if step1 is enemy and step2 is corner
# if so, remove enemy stone
def checkBeatingWithCorner(board, which_team_is_on, to_x, to_y):
    enemy = hu.ATTACKER_PIECE
    beatingOccured = False

    if which_team_is_on == hu.TEAM_ATTACKER:
        enemy = hu.DEFENDER_PIECE

    # up
    if to_x == 2:
        if (
            all(board[to_x - 1, to_y] == enemy)
            & all(board[to_x - 2, to_y] == hu.NO_PIECE)
            & (to_y == 0 | to_y == len(board) - 1)
        ):
            beatingOccured = True
            board[to_x - 1, to_y] = hu.NO_PIECE

    # down
    if to_x == len(board) - 3:
        if (
            all(board[to_x + 1, to_y] == enemy)
            & all(board[to_x + 2, to_y] == hu.NO_PIECE)
            & (to_y == 0 | to_y == len(board) - 1)
        ):
            beatingOccured = True
            board[to_x + 1, to_y] = hu.NO_PIECE

    # left
    if to_y - 2 == 0:
        if (
            all(board[to_x, to_y - 1] == enemy)
            & all(board[to_x, to_y - 2] == hu.NO_PIECE)
            & (to_x == 0 | to_x == len(board) - 1)
        ):
            beatingOccured = True
            board[to_x, to_y - 1] = hu.NO_PIECE

    # right
    if to_y == len(board) - 3:
        if (
            all(board[to_x, to_y + 1] == enemy)
            & all(board[to_x, to_y + 2] == hu.NO_PIECE)
            & (to_x == 0 | to_x == len(board) - 1)
        ):
            beatingOccured = True
            board[to_x, to_y + 1] = hu.NO_PIECE

    return board, beatingOccured


# simple case: our move encapsuled an enemy stone
# check in all 4 directions if step1 is enemy and step2 is friendly
# if so, remove enemy stone
def checkSimpleBeating(board, _which_team_is_on, to_x, to_y):
    enemy = hu.ATTACKER_PIECE
    friendly = hu.DEFENDER_PIECE
    beatingOccured = False

    if _which_team_is_on == hu.TEAM_ATTACKER:
        enemy = hu.DEFENDER_PIECE
        friendly = hu.ATTACKER_PIECE

    # up
    if to_x > 1:
        if all(board[to_x - 1, to_y] == enemy) & all(board[to_x - 2, to_y] == friendly):
            board[to_x - 1, to_y] = hu.NO_PIECE
            beatingOccured = True

    # down
    if to_x < len(board) - 2:
        if all(board[to_x + 1, to_y] == enemy) & all(board[to_x + 2, to_y] == friendly):
            board[to_x + 1, to_y] = hu.NO_PIECE
            beatingOccured = True

    # left
    if to_y > 1:
        if all(board[to_x, to_y - 1] == enemy) & all(board[to_x, to_y - 2] == friendly):
            board[to_x, to_y - 1] = hu.NO_PIECE
            beatingOccured = True

    # right
    if to_y < len(board) - 2:
        if all(board[to_x, to_y + 1] == enemy) & all(board[to_x, to_y + 2] == friendly):
            board[to_x, to_y + 1] = hu.NO_PIECE
            beatingOccured = True

    return board, beatingOccured
