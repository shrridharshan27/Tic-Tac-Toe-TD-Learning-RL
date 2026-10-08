"""
Self-Play Training and Evaluation Pipeline.
"""
from typing import Tuple, Dict
from .game import TicTacToe
from .agent import RLAgent

def train(episodes: int = 5000, alpha: float = 0.5, epsilon: float = 0.2) -> Tuple[RLAgent, RLAgent]:
    """Train two agents via competitive self-play."""
    agent_x = RLAgent(player='X', epsilon=epsilon, alpha=alpha)
    agent_o = RLAgent(player='O', epsilon=epsilon, alpha=alpha)

    for ep in range(episodes):
        game = TicTacToe()
        prev_state_x = None
        prev_state_o = None

        while not game.is_game_over():
            state_before = game.get_state()

            if game.current_player == 'X':
                move = agent_x.choose_move(game)
                prev_state_x = state_before
                game.make_move(move)
                state_after = game.get_state()

                if game.winner == 'X':
                    agent_x.update(prev_state_x, state_after, reward=1.0)
                    if prev_state_o:
                        agent_o.update(prev_state_o, state_after, reward=-1.0)
                elif game.is_draw():
                    agent_x.update(prev_state_x, state_after, reward=0.5)
                    if prev_state_o:
                        agent_o.update(prev_state_o, state_after, reward=0.5)
                else:
                    agent_x.update(prev_state_x, state_after, reward=0.0)

            else:
                move = agent_o.choose_move(game)
                prev_state_o = state_before
                game.make_move(move)
                state_after = game.get_state()

                if game.winner == 'O':
                    agent_o.update(prev_state_o, state_after, reward=1.0)
                    if prev_state_x:
                        agent_x.update(prev_state_x, state_after, reward=-1.0)
                elif game.is_draw():
                    agent_o.update(prev_state_o, state_after, reward=0.5)
                    if prev_state_x:
                        agent_x.update(prev_state_x, state_after, reward=0.5)
                else:
                    agent_o.update(prev_state_o, state_after, reward=0.0)

    return agent_x, agent_o

def evaluate(agent_x: RLAgent, agent_o: RLAgent, games: int = 1000) -> Dict[str, float]:
    """Evaluate trained agents under greedy action selection (epsilon=0)."""
    orig_eps_x = agent_x.epsilon
    orig_eps_o = agent_o.epsilon
    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0

    x_wins = 0
    o_wins = 0
    draws = 0

    for _ in range(games):
        game = TicTacToe()
        while not game.is_game_over():
            if game.current_player == 'X':
                move = agent_x.choose_move(game)
            else:
                move = agent_o.choose_move(game)
            game.make_move(move)

        if game.winner == 'X':
            x_wins += 1
        elif game.winner == 'O':
            o_wins += 1
        else:
            draws += 1

    agent_x.epsilon = orig_eps_x
    agent_o.epsilon = orig_eps_o

    return {
        'x_wins': x_wins,
        'o_wins': o_wins,
        'draws': draws,
        'x_win_pct': (x_wins / games) * 100,
        'o_win_pct': (o_wins / games) * 100,
        'draw_pct': (draws / games) * 100,
    }
