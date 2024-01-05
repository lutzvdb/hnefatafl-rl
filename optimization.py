import torch
import numpy as np

""" Run optimizer on a batch of sequences """


def optimize_batch(optimizer, model_to_optimize, model_target_estimation, batch, gamma):
    (
        boards,
        immediate_reward,
        immediate_reward_opponent,
        not_terminated,
        actions,
        boards_after_opponent_action,
    ) = extract_data_from_batches(batch)

    # Calculate Q prediction target
    discounted_reward_target = calculate_discounted_reward(
        immediate_reward,
        immediate_reward_opponent,
        not_terminated,
        boards_after_opponent_action,
        model_target_estimation,
        gamma,
    )

    # Generate lossy Q estimate
    estimated_from_piece_Q, estimated_to_piece_Q = model_to_optimize.forward(boards)

    # Extract Q estimates for actions actually taken
    estimated_action_q_from = torch.zeros_like(discounted_reward_target)
    estimated_action_q_to = torch.zeros_like(discounted_reward_target)

    # These are the estimated rewards for all possible actions..
    # We only actually did one action
    # Extract that action to calculate the loss
    for i in range(actions.shape[0]):
        estimated_action_q_from[i] = estimated_from_piece_Q[
            i, actions[i, 0, 0], actions[i, 0, 1]
        ]
        estimated_action_q_to[i] = estimated_to_piece_Q[
            i, actions[i, 1, 0], actions[i, 1, 1]
        ]

    estimated_action_q = (estimated_action_q_from + estimated_action_q_to) / 2

    # Now we can finally compute the actual loss
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(estimated_action_q, discounted_reward_target)

    # And do the optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


""" Derive discounted reward (prediction target) """


def calculate_discounted_reward(
    immediate_reward,
    immediate_reward_opponent,
    not_terminated,
    boards_after_opponent_action,
    model_target_estimation,
    gamma,
):
    # Estimate Q-value for next state for non-terminated cases
    estimated_next_reward_from_max_q = torch.zeros_like(immediate_reward)
    estimated_next_reward_to_max_q = torch.zeros_like(immediate_reward)

    if torch.any(not_terminated):
        board_next_move = torch.tensor(
            boards_after_opponent_action[not_terminated], dtype=torch.float32
        )
        q_from, q_to = model_target_estimation.forward(board_next_move)
        estimated_next_reward_from_max_q[not_terminated] = torch.amax(
            q_from, dim=(1, 2)
        )
        estimated_next_reward_to_max_q[not_terminated] = torch.amax(q_to, axis=(1, 2))

    # Both from and to should estimate the same reward
    estimated_next_reward = (
        estimated_next_reward_from_max_q + estimated_next_reward_to_max_q
    ) / 2

    # Big caveat: We are not adding the discounted estimated reward for the opponent
    discounted_reward_target = (
        immediate_reward - immediate_reward_opponent + gamma * estimated_next_reward
    )

    return discounted_reward_target


""" Reshape data from batches """


def extract_data_from_batches(batch):
    # Batches consist of entries of the form:
    # [state, action, reward, next_state, terminated, boards_after_opponent_action]
    # Extract relevant vectors from batch entries
    # Because we have n number of environments, the true batch size is
    # num_envs * batch_size
    # or len(batch_size) * len(batch_size[0])
    board_size = batch[0][0].shape[1]

    boards = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
    boards = boards.reshape((-1, board_size, board_size, 3))
    immediate_reward = torch.tensor(
        np.array([b[2] for b in batch]), dtype=torch.float32
    ).flatten()
    immediate_reward_opponent = torch.tensor(
        np.array([b[6] for b in batch]), dtype=torch.float32
    ).flatten()
    not_terminated = torch.logical_not(
        torch.tensor(np.array([b[4] for b in batch]), dtype=torch.bool).flatten()
    )
    actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.int8)
    boards_after_opponent_action = torch.tensor(
        np.array([b[5] for b in batch]), dtype=torch.float32
    )
    # Reshape: We have (BATCH_SIZE, n_env, board_size, board_size, 3) entries
    # We want (BATCH_SIZE * n_env, board_size, board_size, 3) entries
    boards_after_opponent_action = boards_after_opponent_action.reshape(
        (-1, board_size, board_size, 3)
    )
    actions = actions.reshape((-1, 2, 2))  # (true_batch_size, from/to, x/y)

    return (
        boards,
        immediate_reward,
        immediate_reward_opponent,
        not_terminated,
        actions,
        boards_after_opponent_action,
    )
