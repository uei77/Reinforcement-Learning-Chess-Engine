import torch
import torch.optim as optim
import os
from neural_network import chess_neural_network
from enginevsbot import manager_vs_bot

ITERATIONS = 5        
GAMES_PER_ITERATION = 5
MCTS_SIMULATIONS = 800    
LEARNING_RATE = 0.001
FINAL_MODEL_NAME = "final_chess_rl_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = chess_neural_network().to(device)
    
    if os.path.exists(FINAL_MODEL_NAME):
        print(f"Checkpoint found: Loading {FINAL_MODEL_NAME}...")
        try:
            model.load_state_dict(torch.load(FINAL_MODEL_NAME, map_location=device), strict=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print(f"Training starts: {ITERATIONS} iterations, {GAMES_PER_ITERATION} games per iteration.")
    
    manager_vs_bot(
        model=model,
        optimizer=optimizer,
        device=device,
        iterations=ITERATIONS,
        games_per_iteration=GAMES_PER_ITERATION,
        simulation=MCTS_SIMULATIONS
    )
    
    torch.save(model.state_dict(), FINAL_MODEL_NAME)
    print("\nTraining is over.")

if __name__ == "__main__":
    main()