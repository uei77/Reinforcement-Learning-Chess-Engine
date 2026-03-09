import torch
import torch.nn as nn
import torch.optim as optim
import os
from trainwithdataset import train_loop
from neural_network import chess_neural_network

CSV_FILE = "games.csv"  
EPOCHS = 10            
BATCH_SIZE = 256  
LEARNING_RATE = 0.001
MODEL_NAME = "final_chess_rl_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Supervised Training: {CSV_FILE} for {EPOCHS} epochs.")
    print(f"Active Device: {device.type.upper()}")
    model = chess_neural_network()
    if os.path.exists(MODEL_NAME):
        print(f"Checkpoint found. Loading {MODEL_NAME}...")
        try:
            checkpoint = torch.load(MODEL_NAME, map_location=device, weights_only=True)
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded successfully. Resuming training.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected. DataParallel activated.")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    try:
        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Starting Epoch {epoch}/{EPOCHS} ---")
            train_loop(
                model=model,
                optimizer=optimizer,
                csv_file=CSV_FILE, 
                epochs=1,
                batch_size=BATCH_SIZE,
                model_path=MODEL_NAME
            )
            print(f"Epoch {epoch} finished. Saving model...")
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, MODEL_NAME)
        print("\nDataset training completed successfully.")
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please verify the file path.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
if __name__ == "__main__":
    main()