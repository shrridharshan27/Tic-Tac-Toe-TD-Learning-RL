"""
Reinforcement Learning Agent for Tic-Tac-Toe using TD(0) Learning.
"""
import random
from typing import Dict, Optional
from .game import TicTacToe

class RLAgent:
    def __init__(self, player: str, epsilon: float = 0.2, alpha: float = 0.5, gamma: float = 0.9):
        """
        Initialize the RL Agent.
        :param player: 'X' or 'O'
        :param epsilon: Exploration rate (probability of random move)
        :param alpha: Learning rate (step-size parameter)
        :param gamma: Discount factor for future state values
        """
        self.player = player
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # V: Value table mapping board state string -> estimated state value
        self.V: Dict[str, float] = {}

    def get_value(self, state: str) -> float:
        """Retrieve estimated value of state; default to neutral 0.5 if unseen."""
        return self.V.get(state, 0.5)

    def choose_move(self, game: TicTacToe) -> int:
        """Select action using epsilon-greedy action selection."""
        empty = game.get_empty_cells()
        if not empty:
            raise ValueError("No empty cells available on the board.")

        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(empty)

        # Exploitation: 1-step lookahead greedy choice
        best_move = None
        best_value = -float('inf')

        for pos in empty:
            game.board[pos] = self.player
            state = game.get_state()
            val = self.get_value(state)
            game.board[pos] = ' '  # Undo simulation

            if val > best_value:
                best_value = val
                best_move = pos

        return best_move if best_move is not None else random.choice(empty)

    def update(self, prev_state: str, next_state: str, reward: float) -> None:
        """
        Temporal Difference (TD) Learning update rule:
        V(s) <- V(s) + alpha * [reward + gamma * V(s') - V(s)]
        """
        old_val = self.get_value(prev_state)
        next_val = self.get_value(next_state)

        td_target = reward + self.gamma * next_val
        td_error = td_target - old_val

        self.V[prev_state] = old_val + self.alpha * td_error
