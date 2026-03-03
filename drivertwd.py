import torch
import torch.optim as optim
import os
from trainwithdataset import train_loop
from neural_network import chess_neural_network

CSV_FILE = "games.csv"  
EPOCHS = 10            
BATCH_SIZE = 128         
LEARNING_RATE = 0.001
MODEL_NAME = "final_chess_dl_model.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Supervised Training: {CSV_FILE} for {EPOCHS} epochs.")
    model = chess_neural_network().to(device)
    if os.path.exists(MODEL_NAME):
        print(f"Checkpoint found! Loading weights from {MODEL_NAME} to continue training...")
        try:
            checkpoint = torch.load(MODEL_NAME, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("Model weights loaded successfully! Resuming training...")
            
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print(f"No checkpoint found ({MODEL_NAME}). Starting training from scratch.")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    try:
       
        train_loop(
            model=model,
            optimizer=optimizer,
            csv_file=CSV_FILE, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
            model_path=MODEL_NAME
        )
        print("\nDataset training completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please verify the file path.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()