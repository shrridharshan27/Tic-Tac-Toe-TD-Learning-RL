"""
Tic-Tac-Toe Environment
Represents a 3x3 game board supporting two players ('X' and 'O').
"""
from typing import List, Optional

class TicTacToe:
    def __init__(self):
        # Board = list of 9 cells, all empty at start
        self.board: List[str] = [' '] * 9
        # X always goes first
        self.current_player: str = 'X'
        # Winner status: 'X', 'O', or None
        self.winner: Optional[str] = None

    def clone(self) -> 'TicTacToe':
        new_game = TicTacToe()
        new_game.board = list(self.board)
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        return new_game

    def get_empty_cells(self) -> List[int]:
        """Return positions (0-8) where board is still empty."""
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, position: int) -> Optional[str]:
        """Place current player's mark at given position and switch turns."""
        self.board[position] = self.current_player
        self.winner = self.check_winner()
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.winner

    def check_winner(self) -> Optional[str]:
        """Check all 8 winning combinations (rows, columns, diagonals)."""
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        return None

    def is_draw(self) -> bool:
        """Draw = no empty cells left AND no winner."""
        return len(self.get_empty_cells()) == 0 and self.winner is None

    def is_game_over(self) -> bool:
        """Game over if someone won OR it is a draw."""
        return self.winner is not None or self.is_draw()

    def print_board(self) -> None:
        """Print board in 3x3 grid style."""
        b = self.board
        print(f" {b[0]} | {b[1]} | {b[2]} ")
        print("---|---|---")
        print(f" {b[3]} | {b[4]} | {b[5]} ")
        print("---|---|---")
        print(f" {b[6]} | {b[7]} | {b[8]} ")

    def get_state(self) -> str:
        """Convert board list to a 9-char string identifier."""
        return ''.join(self.board)
