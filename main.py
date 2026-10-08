"""
Main Execution Script for Tic-Tac-Toe TD Learning
Shrri Dharshan D R - 23BPS1090
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tictactoe.game import TicTacToe
from tictactoe.agent import RLAgent
from tictactoe.train import train, evaluate

def play_vs_agent(agent_x: RLAgent) -> None:
    """Interactive human vs agent gameplay."""
    print("\n--- Play against the trained agent! ---")
    print("You are O. Agent is X.")
    print("Board positions:")
    print(" 1 | 2 | 3 \n---|---|---\n 4 | 5 | 6 \n---|---|---\n 7 | 8 | 9 ")

    agent_x.epsilon = 0.0
    game = TicTacToe()

    while not game.is_game_over():
        game.print_board()
        if game.current_player == 'X':
            print("\nAgent (X) is thinking...")
            move = agent_x.choose_move(game)
            game.make_move(move)
            print(f"Agent placed X at position {move + 1}")
        else:
            empty = [i + 1 for i in game.get_empty_cells()]
            print(f"\nYour turn (O). Available: {empty}")
            while True:
                try:
                    choice = int(input("Enter position (1-9): "))
                    if choice in empty:
                        game.make_move(choice - 1)
                        break
                    print("Invalid move! Choose from:", empty)
                except ValueError:
                    print("Please enter a valid number!")

    game.print_board()
    if game.winner == 'X':
        print("\nAgent wins! Optimal TD policy proved triumphant.")
    elif game.winner == 'O':
        print("\nYou win! Well played.")
    else:
        print("\nIt\'s a draw! Perfect defensive strategy by both players.")

def run_experiment_and_plot():
    print("=" * 60)
    print("   REINFORCEMENT LEARNING LAB 1: TIC-TAC-TOE TD LEARNING")
    print("=" * 60)

    episodes_list = [1000, 5000, 10000, 15000]
    x_wins_list = []
    o_wins_list = []
    draws_list = []
    brain_sizes_x = []

    for ep in episodes_list:
        print(f"\nTraining self-play agents for {ep} episodes...")
        agent_x, agent_o = train(episodes=ep)
        res = evaluate(agent_x, agent_o, games=1000)
        x_wins_list.append(res['x_win_pct'])
        o_wins_list.append(res['o_win_pct'])
        draws_list.append(res['draw_pct'])
        brain_sizes_x.append(len(agent_x.V))
        print(f"  Results ({ep} episodes): X Wins={res['x_win_pct']:.1f}%, O Wins={res['o_win_pct']:.1f}%, Draws={res['draw_pct']:.1f}%")
        print(f"  Agent X state space explored: {len(agent_x.V)} states")

    # Generate Performance Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(episodes_list, x_wins_list, marker='o', color='#2b5c8f', lw=2, label='X Wins % (First Player)')
    axes[0].plot(episodes_list, o_wins_list, marker='s', color='#d9534f', lw=2, label='O Wins % (Second Player)')
    axes[0].plot(episodes_list, draws_list, marker='^', color='#5cb85c', lw=2, label='Draws %')
    axes[0].set_title('Outcome Distribution vs Training Episodes', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Training Episodes', fontsize=11)
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].plot(episodes_list, brain_sizes_x, marker='D', color='#8a2be2', lw=2)
    axes[1].set_title('Agent X State-Value Coverage (Brain Size)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Training Episodes', fontsize=11)
    axes[1].set_ylabel('Unique States Explored', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("assets", "training_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    run_experiment_and_plot()
