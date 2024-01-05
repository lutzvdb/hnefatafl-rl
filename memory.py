import random

"""
Simple replay memory class, storing transitions and sampling from them.
Non-standard is the option to alter memory after the fact to allow for 
knowing the state after the opponent action
which is necessary for discounted rewards.
"""


class ReplayMemory:
    def __init__(self, n_remember=10000):
        self.memory = []
        self.n_remember = n_remember

    """ Add a sequence to memory  """

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        boards_after_opponent_action,
        enemy_reward_after_action,
    ):
        self.memory.append(
            [
                state,
                action,
                reward,
                next_state,
                terminated,
                boards_after_opponent_action,
                enemy_reward_after_action,
            ]
        )
        if len(self.memory) > self.n_remember:
            del self.memory[0]

    """ Update the last-added memory item with boards_after_opponent_action """

    def update_memory_after_opponent_action(
        self, boards_after_opponent_action, enemy_reward_after_action
    ):
        if len(self.memory) == 0:
            return

        self.memory[len(self.memory) - 1][5] = boards_after_opponent_action
        self.memory[len(self.memory) - 1][6] = enemy_reward_after_action

    """ Get (decorrelated) samples from memory """

    def sample(self, n=None):
        if n == None:
            # Sample all
            return self.memory

        if n > len(self.memory):
            return self.memory

        # sample n memory items
        return random.sample(self.memory, n)
