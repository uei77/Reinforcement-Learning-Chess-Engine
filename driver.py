import torch
import torch.optim as optim
from neural_network import chess_neural_network
import os
from train_with_playing_itself import manager

ITERATIONS = 5        
GAMES_PER_ITERATION = 5
MCTS_SIMULATIONS = 400     
LEARNING_RATE = 0.001
FINAL_MODEL_NAME="final_chess_dl_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = chess_neural_network().to(device)
    if os.path.exists(FINAL_MODEL_NAME):
        print(f"Checkpoint found: Loading {FINAL_MODEL_NAME}...")
        try:
            model.load_state_dict(torch.load(FINAL_MODEL_NAME, map_location=device), strict=False)
            print("Model loaded successfully! Starting self-play with pre-trained knowledge.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch (Random weights).")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    print(f"Training starts: {ITERATIONS}  Games per iteration:{GAMES_PER_ITERATION} ")
    manager(model=model,
            optimizer=optimizer,
            device=device,
            iterations=ITERATIONS,
            games_per_iteration=GAMES_PER_ITERATION,
            simulation=MCTS_SIMULATIONS
            )
    
    torch.save(model.state_dict(), FINAL_MODEL_NAME)
    print("\n Training is over ")

if __name__ == "__main__":
    main()
    