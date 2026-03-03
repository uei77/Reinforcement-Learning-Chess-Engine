import torch
import torch.nn as nn
import torch.optim as optim
import gc
import os
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import chess
import numpy as np
from torch.amp import autocast, GradScaler
from neural_network import chess_neural_network 
from chess_board import board_to_tensor
from move import Move

class Train(Dataset):
    def __init__(self, csvfile, max_samples=None):
        self.data = pd.read_csv(csvfile, usecols=['moves', 'winner'], nrows=max_samples)
        self.move_encoder = Move()
        print(f"{len(self.data)} games has been uploaded. Processing")
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        board, next_move_notation = self.get_random_game_state(row['moves'])
        if board is None:
            return self.__getitem__((index + 1) % len(self.data))
        try:
            policy_target, value_target = self.targets(board, next_move_notation, row['winner'])
            board_tensor = board_to_tensor(board)
            return board_tensor, policy_target, value_target
        except Exception:
            return self.__getitem__((index + 1) % len(self.data))
    
    def get_random_game_state(self, notation):
        moves = notation.split()
        if len(moves) == 0:
            return None, None
        try:
            random_index = np.random.randint(0, len(moves))
        except ValueError:
            return None, None
        board = chess.Board()
        for i in range(random_index):
            board.push_san(moves[i])
        
        next_move_notation = moves[random_index]
        return board, next_move_notation
        
    def targets(self, board, next_move_notation, winner):
        next_move = board.parse_san(next_move_notation)
        policy_index = self.move_encoder._encode_move(next_move)
        
        if policy_index is None:
            raise ValueError("Cannot encode move")
        if winner == 'white':
            game_result = 1.0
        elif winner == 'black':
            game_result = -1.0
        else:
            game_result = 0.0
            
        if board.turn == chess.BLACK:
            value_target = -game_result
        else:
            value_target = game_result
            
        return (
            torch.tensor(policy_index, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32)
        )
        
def train_batch(model, data, optimizer, criterion_policy, criterion_value, device, scaler):
    boards, policy_targets, value_targets = data
    boards = boards.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device).unsqueeze(1)
    
    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float16):
        prediction_value, prediction_policy = model(boards)
        loss_value = criterion_value(prediction_value, value_targets)
        loss_policy = criterion_policy(prediction_policy, policy_targets)
        total_loss = loss_value + loss_policy
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss.item(), loss_policy.item(), loss_value.item()
    
def save_checkpoint(model, epoch, filename_prefix="chess_model"):
    filename = f"final_chess_dl_model.pt"
    torch.save(model.state_dict(), filename)
    print(f"Model has been saved: {filename}")
        
def train_loop(model, optimizer, csv_file, epochs=10, batch_size=128, model_path="final_chess_dl_model.pt"):
    device = next(model.parameters()).device
    print(f"Training started Device: {device}")
    scaler = GradScaler(device.type)
    dataset = Train(csv_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Total positions: {len(dataset)} | Train: {train_size} | Validation: {val_size}")
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 3 
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        print(f"\nEpoch {epoch}/{epochs} Starts")
        for i, batch_data in enumerate(train_loader):
            loss, p_loss, v_loss = train_batch(model, batch_data, optimizer, criterion_policy, criterion_value, device, scaler)
            epoch_loss += loss
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {loss:.4f} (Policy:{p_loss:.3f} Value:{v_loss:.3f})")
                
        avg_train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                boards, policy_targets, value_targets = batch_data
                boards = boards.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device).unsqueeze(1)
                
                with autocast(device_type=device.type, dtype=torch.float16):
                    prediction_value, prediction_policy = model(boards)
                    l_v = criterion_value(prediction_value, value_targets)
                    l_p = criterion_policy(prediction_policy, policy_targets)
                    val_loss += (l_v + l_p).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} ends. Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model has been safely saved: {model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(" OVERFITTING DETECTED: Model is memorizing the dataset. Early stopping triggered.")
                break
            
if __name__ == "__main__":
    train_loop("games.csv", epochs=10, batch_size=128)