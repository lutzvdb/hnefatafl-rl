# Simple reinforcement learning algorithm for hnefatafl

## Overview

This is an attempt at an implementation of a RL-algorithm for the ancient viking game hnefatafl, as played online on hnefatafl.app. It's pretty barebones and was mostly meant as an excercise in applying newly learnt RL techniques to an interesting problem.

If you're interested in collaborating on this, please reach out.

## Open issues and questions

- Resulting gameplay can't really be considered intelligent.
- I'm unsure about how to define the model architecture. For now I'm using a mixture of first convolutional layers and then fully connected ones, however I'm not sure that that's a good approach.
- I am jointly estimating the estimated Q-value for which piece to pick (read: average Q value for moving this piece) and where to put it (read: average Q value of moving any piece to this position). This is not optimal, as a single target location obviously can have different true Q values depending on what piece was moved here. Again, I'm unsure on how to better model this. Maybe two distinct networks - one for from and one for to, given the board and the picked from piece as input?
- Strong bias towards defender under current training
- No MCTS involved, just an estimate of discounted reward. MCTS similar to AlphaGo could make sense.
- No discounted (t+1) opponent reward is added to loss, only the own (t+1) reward. A difficulty lies in either having to sample an action for calculating opponent reward (which is noisy) or having to calculate all rewards for all possible actions and averaging them (which is computationally intensive).
