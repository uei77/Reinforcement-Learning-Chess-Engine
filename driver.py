import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import chess_neural_network
import os
import chess.polyglot
from train_with_playing_itself import manager

ITERATIONS = 15      
GAMES_PER_ITERATION = 10
MCTS_SIMULATIONS = 800    
LEARNING_RATE = 0.001
FINAL_MODEL_NAME = "final_chess_rl_model.pt"
OPENING_BOOK_PATH = "Perfect2021.bin"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device.type.upper()}")
    
    if os.path.exists(OPENING_BOOK_PATH):
        print(f"Opening book found: {OPENING_BOOK_PATH}")
    else:
        print("Opening book not found.")
        
    model = chess_neural_network()
    
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected. DataParallel activated.")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    if os.path.exists(FINAL_MODEL_NAME):
        print(f"Checkpoint found. Loading {FINAL_MODEL_NAME}...")
        try:
            checkpoint = torch.load(FINAL_MODEL_NAME, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully. Starting self-play with pre-trained knowledge.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    print(f"Training starts. Iterations: {ITERATIONS} | Games per iteration: {GAMES_PER_ITERATION}")
    
    manager(
        model=model,
        optimizer=optimizer,
        device=device,
        iterations=ITERATIONS,
        games_per_iteration=GAMES_PER_ITERATION,
        simulation=MCTS_SIMULATIONS,
        opening_book_path=OPENING_BOOK_PATH
    )
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), FINAL_MODEL_NAME)
    else:
        torch.save(model.state_dict(), FINAL_MODEL_NAME)
    print("\nTraining is over.")

if __name__ == "__main__":
    main()