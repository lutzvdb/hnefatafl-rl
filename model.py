import torch
import torch.nn.functional as F
import numpy as np


class ConvAndNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Policy(torch.nn.Module):
    def __init__(self, board_sidelength=7, hidden_size=64):
        super().__init__()
        self.board_sidelength = board_sidelength
        self.hidden_size = hidden_size
        full_pad = int((board_sidelength - 1) // 2)

        self.base_conv1 = ConvAndNorm(
            in_channels=3, out_channels=hidden_size // 4, kernel_size=3, padding=1
        )
        self.base_conv2 = ConvAndNorm(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 2,
            kernel_size=5,
            padding=2,
        )
        self.base_conv3 = ConvAndNorm(
            in_channels=hidden_size // 2,
            out_channels=hidden_size,
            kernel_size=board_sidelength,
            padding=full_pad,
        )
        self.fc_from = torch.nn.Linear(
            in_features=hidden_size * board_sidelength * board_sidelength,
            out_features=board_sidelength * board_sidelength,
        )
        self.fc_to = torch.nn.Linear(
            in_features=hidden_size * board_sidelength * board_sidelength,
            out_features=board_sidelength * board_sidelength,
        )

    def forward(self, board):
        # conv's are channels-first
        # switch from (batch, x, y, channel)
        # to
        # (batch, channel, x, y)
        x = torch.permute(board, (0, 3, 1, 2))
        # Identical preprocessing for "to" and "from"
        x = F.relu(self.base_conv1(x))
        x = F.relu(self.base_conv2(x))
        x = F.relu(self.base_conv3(x))
        x = x.view(
            (-1, self.hidden_size * self.board_sidelength * self.board_sidelength)
        )

        # Seperate outputs for from and to
        # These Q values represent the estimated discounted reward for each action
        from_piece_Q = self.fc_from(x).view(
            -1, self.board_sidelength, self.board_sidelength
        )
        to_piece_Q = self.fc_to(x).view(
            -1, self.board_sidelength, self.board_sidelength
        )

        return from_piece_Q, to_piece_Q
