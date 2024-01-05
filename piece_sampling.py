import torch
import torch.nn.functional as F
import numpy as np
import hnefatafl_utils as hu
import ruleset as rules

""" Convert (masked) q values to probabiltiies  """


def q_values_to_probabilities(q_values, temperature):
    board_sidelength = q_values.shape[1]
    prep = q_values.view(-1, board_sidelength**2) / temperature
    probs = F.softmax(prep, dim=-1)
    # If all values into softmax are -inf (which can hapapen if no moves are possible),
    # we have to replace the probs (all nan) with 0
    probs = torch.nan_to_num(probs, 0)
    probs = probs.view(-1, board_sidelength, board_sidelength)

    return probs


""" Turn Q-values from network into sampled actions (from-piece and to-piece) """


def generate_actions_from_q(
    from_piece_Q, to_piece_Q, board, which_team_is_on, temperature
):
    # Mask the outputs, pick pieces and move
    # It's possible that we pick a from_piece that is unable to move (e.g. king as 1st move)
    # In that case, pick again
    batch_size = from_piece_Q.shape[0]
    # only re-sample for invalid cases; keep track of indexes here
    mis_idx = torch.tensor(range(batch_size), dtype=torch.int)
    picked_from_pieces = torch.tensor(np.repeat([-1, -1], batch_size), dtype=torch.int8)
    picked_from_pieces = picked_from_pieces.reshape(batch_size, 2)

    to_distr_masked = torch.zeros_like(from_piece_Q)

    # First we mask out nonsensical moves, then softmax for probabilities
    from_q_masked = apply_from_mask_batched(from_piece_Q, board, which_team_is_on)
    from_distr_masked = q_values_to_probabilities(from_q_masked, temperature)

    emergency_cnt = 0
    while len(mis_idx) > 0:
        emergency_cnt += 1
        # (1) Sample a from piece
        picked_from_pieces[mis_idx] = sample_from_piece_distribution_batched(
            from_distr_masked[mis_idx]
        )
        # (2) Mask possible to locations
        to_distr_masked[mis_idx] = apply_to_mask_batched(
            to_piece_Q[mis_idx], board[mis_idx], picked_from_pieces[mis_idx]
        )
        # (3) Turn masked Q-values into to-probabilities
        to_distr_masked[mis_idx] = q_values_to_probabilities(
            to_distr_masked[mis_idx], temperature
        )

        # (4) If we have picked a piece in (1) that has nowhere to go,
        #     set the probability for that from piece to zero
        to_distr_sum = torch.sum(to_distr_masked, axis=(-1, -2))
        mis_idx = torch.where(to_distr_sum == 0)[0]

        for i in range(len(mis_idx)):
            # Ensure that we don't re-pick this figure
            # by setting its from_probability to 0
            from_q_masked[
                mis_idx[i],
                picked_from_pieces[mis_idx[i], 0].squeeze(),
                picked_from_pieces[mis_idx[i], 1].squeeze(),
            ] = -torch.inf
            from_distr_masked = q_values_to_probabilities(from_q_masked, temperature)

        if (torch.sum(from_distr_masked[mis_idx]) <= 0) & (len(mis_idx) > 0):
            print("No figures left; player has lost?")
            print(from_distr_masked[mis_idx])
            print(board[mis_idx])
            break

        if emergency_cnt > 100:
            print(
                "Emergency break. This should not happen. Check stopping criterion in generate_actions_from_q()."
            )
            break

    # Pick to pieces based on derived to probabilities
    picked_to_pieces = sample_from_piece_distribution_batched(to_distr_masked)

    actions = [
        (picked_from_pieces[i], picked_to_pieces[i])
        for i in range(len(picked_from_pieces))
    ]

    return actions


""" Apply from-piece mask to a batched tensor """


def apply_from_mask_batched(to_be_masked, board, which_team_is_on):
    # Iterate through boards / masks --> do it for every item in batch
    for i in range(to_be_masked.shape[0]):
        to_be_masked[i] = apply_from_mask(to_be_masked[i], board[i], which_team_is_on)

    return to_be_masked


""" Apply a mask that zeroes out all entries in a given board that
    are not belonging to the current team """


def apply_from_mask(to_be_masked, board, which_team_is_on):
    """def apply_single_from_mask(to_be_masked, board, which_team_is_on):"""
    if which_team_is_on == hu.TEAM_ATTACKER:
        relevant_pieces = torch.tensor(
            list(
                map(
                    lambda arr: list(map(torch.all, arr)),
                    board == torch.tensor(hu.ATTACKER_PIECE, dtype=torch.float32),
                )
            ),
            dtype=torch.float32,
        )
    else:
        defender_pieces = torch.tensor(
            list(
                map(
                    lambda arr: list(map(torch.all, arr)),
                    board == torch.tensor(hu.DEFENDER_PIECE, dtype=torch.float32),
                )
            ),
            dtype=torch.float32,
        )
        king_piece = torch.tensor(
            list(
                map(
                    lambda arr: list(map(torch.all, arr)),
                    board == torch.tensor(hu.KING_PIECE, dtype=torch.float32),
                )
            ),
            dtype=torch.float32,
        )
        relevant_pieces = defender_pieces + king_piece

    # 0 means irrelevant, 1 relevant
    # Transform to -inf for irrelevant pieces and 0 for relevant
    relevant_pieces = relevant_pieces - 1
    relevant_pieces[relevant_pieces == -1] = -torch.inf
    res = torch.add(to_be_masked, relevant_pieces)

    return res


""" Sample a single piece from a distribution of pieces, batched """


def sample_from_piece_distribution_batched(distribution, seed=None):
    picked_pieces = []
    board_sidelength = distribution.shape[1]
    batch_size = distribution.shape[0]
    if seed:
        torch.random.manual_seed(seed)

    # Could probably be made more efficient
    # seeing how torch.multinomial can handle batching (one row per batch item)

    # Sample per batch
    for i in range(batch_size):
        x = distribution[i].flatten()
        if torch.sum(x) <= 0:
            # Invalid probabilities: No actions possible
            picked_pieces.append([-1, -1])
            continue

        picked_idx = torch.multinomial(x, 1)
        picked_pieces.append(
            [int(picked_idx // board_sidelength), int(picked_idx % board_sidelength)]
        )
    picked_pieces = torch.tensor(picked_pieces, dtype=torch.int8)

    return picked_pieces


def apply_to_mask_batched(to_be_masked, board, from_piece):
    # Iterate through boards / masks --> do it for every item in batch
    for i in range(to_be_masked.shape[0]):
        to_be_masked[i] = apply_to_mask(to_be_masked[i], board[i], from_piece[i])

    return to_be_masked


"""  Returns a mask with valid moves for a given piece """


def apply_to_mask(to_be_masked, board, from_piece):
    # (1) We can go only vertically and horizontally
    # (2) We cannot go on the same spot as other players
    # (3) We cannot go over other players
    # (4) We cannot go into the corners or the throne - except as king
    # (5) The king can only do max 3 steps

    from_x = from_piece[0]
    from_y = from_piece[1]
    picked_piece = board[from_x, from_y]

    blank_mask = torch.zeros((board.shape[0], board.shape[0]))
    # We go the 4 possible directions until we hit a problem

    def walk_direction(mask, board, picked_piece, from_piece, direction_vector):
        cur_pos = from_piece
        direction_vector = torch.tensor(direction_vector, dtype=torch.int8)
        walked_distance = torch.tensor([0], dtype=torch.float32)
        we_are_king = torch.all(
            picked_piece == torch.tensor(hu.KING_PIECE, dtype=torch.float32)
        )
        board_len = board.shape[0] - 1
        while True:
            cur_pos = cur_pos + direction_vector
            walked_distance = walked_distance + torch.sum(direction_vector)
            if any(cur_pos > board_len) | any(cur_pos < 0):
                # Don't fall off the board
                break
            if we_are_king == True:
                if walked_distance > 3:
                    break  # (5)
            if torch.any(
                board[cur_pos[0], cur_pos[1]]
                != torch.tensor(hu.NO_PIECE, dtype=torch.float32)
            ):
                break  # we hit a roadblock - (3) and (2)

            if (we_are_king == False) & (
                torch.equal(cur_pos, torch.tensor([0, 0], dtype=torch.float32))
                | torch.equal(
                    cur_pos, torch.tensor([0, board_len], dtype=torch.float32)
                )
                | torch.equal(
                    cur_pos, torch.tensor([board_len, 0], dtype=torch.float32)
                )
                | torch.equal(
                    cur_pos, torch.tensor([0, board_len], dtype=torch.float32)
                )
            ):
                # Corner as non-king
                break  # (4)
            if (we_are_king == False) & torch.equal(
                cur_pos,
                torch.tensor([board_len // 2, board_len // 2], dtype=torch.float32),
            ):
                # Throne as non-king
                break  # (4)

            # all rules are met, allow this step
            mask[cur_pos[0], cur_pos[1]] = 1

        return mask

    mask = walk_direction(blank_mask, board, picked_piece, from_piece, [1, 0])  # down
    mask = walk_direction(mask, board, picked_piece, from_piece, [0, 1])  # right
    mask = walk_direction(mask, board, picked_piece, from_piece, [-1, 0])  # up
    mask = walk_direction(mask, board, picked_piece, from_piece, [0, -1])  # left

    # 0 means irrelevant, 1 relevant
    # Transform to -inf for irrelevant pieces and 0 for relevant
    mask = mask - 1
    mask[mask == -1] = -torch.inf
    to_be_masked = torch.add(to_be_masked, mask)

    return to_be_masked
