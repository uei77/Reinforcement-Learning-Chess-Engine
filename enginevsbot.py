import torch
from torch.amp import GradScaler, autocast
import os
import chess
import chess.polyglot
import numpy as np
import random
import copy
from move import Move
from chess_board import board_to_tensor
from monte_carlo_search_tree import monte_carlo_search_algorithm

move_record = Move()

def get_simple_python_bot_move(board):
    legal_moves = list(board.legal_moves)
    best_move = random.choice(legal_moves)
    
    for move in legal_moves:
        if board.is_capture(move):
            best_move = move
            break 
            
    return best_move

def play_move_with_book(board, model, device, simulation=200, is_Train=True, move_count=0):
    try:
        with chess.polyglot.open_reader("Perfect2021.bin") as reader:
            entry = reader.choice(board)
            book_move = entry.move
            move_policy = np.zeros(4672)
            idx = move_record._encode_move(book_move)
            if idx is not None:
                move_policy[idx] = 1.0
            return book_move, move_policy
    except IndexError:
        pass
    except FileNotFoundError:
        pass
        
    move, move_policy = monte_carlo_search_algorithm(board, model, device, numofsimulation=simulation, is_Train=is_Train, move_count=move_count)
    return move, move_policy

def get_game_result(board):
    res = board.result()
    if res == "1-0": return 1.0 
    if res == "0-1": return -1.0
    return 0.0

def prepare_data(history, result):
    states = []
    policies = []
    values = []
    reward = result
    for board_state, policy_vector in history:
        states.append(board_to_tensor(board_state))
        policies.append(torch.tensor(policy_vector, dtype=torch.float32))
        values.append(torch.tensor([reward], dtype=torch.float32))
        reward = -reward
    return states, policies, values

def process_game_vs_bot(model, device, game_idx, simulation=200):
    model.eval()
    board = chess.Board()
    game_history = []
    
    model_is_white = random.choice([True, False])
    print(f"Game {game_idx+1} starts. DL Model is {'WHITE' if model_is_white else 'BLACK'}")
    
    with torch.no_grad():
        while not board.is_game_over() and len(board.move_stack) < 150:
            current_move_count = len(board.move_stack)
            
            if (board.turn == chess.WHITE and model_is_white) or (board.turn == chess.BLACK and not model_is_white):
                with autocast(device_type=device.type, dtype=torch.float16):
                    move, policy = play_move_with_book(board, model, device, simulation=simulation, move_count=current_move_count)
                if move is None:
                    print(f"Warning: Game {game_idx+1} terminated early. No MCTS move found.")
                    break
            else:
                move = get_simple_python_bot_move(board)
                policy = np.zeros(4672)
                idx = move_record._encode_move(move)
                if idx is not None:
                    policy[idx] = 1.0
            
            game_history.append((board.copy(), policy))
            board.push(move)
            
    final_result = get_game_result(board)
    if not board.is_game_over() and len(board.move_stack) >= 150:
        final_result = 0.0 
        
    states, policies, values = prepare_data(game_history, final_result)
    print(f"Game {game_idx+1} is over. Result: {final_result} | Total Moves: {len(board.move_stack)}")
    return states, policies, values

def train_with_playing_itself_buffer(model, optimizer, states, policies, values, device, epochs=1, scaler=None):
    model.train()
    state_tensor = torch.stack(states).to(device)
    policy_targets = torch.stack(policies).to(device)
    value_targets = torch.stack(values).to(device)
    loss_sum = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.float16):
            value_prediction, policy_prediction = model(state_tensor)
            loss_value = torch.nn.functional.mse_loss(value_prediction.view(-1, 1), value_targets.view(-1, 1))
            log_softmax_p = torch.nn.functional.log_softmax(policy_prediction, dim=1)
            loss_policy = -torch.mean(torch.sum(policy_targets * log_softmax_p, dim=1))
            total_loss = loss_value + loss_policy 
            
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
            
        loss_sum += total_loss.item()
        print(f"Train Epoch {epoch+1}/{epochs} | Total Loss: {total_loss.item():.4f}")
        
    return loss_sum / epochs

def manager_vs_bot(model, optimizer, device, iterations=10, games_per_iteration=5, simulation=200):
    model.to(device)
    scaler = GradScaler(device.type)
    schedule_tracker = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if os.path.exists("final_chess_dl_model.pt"):
        print("Loading existing model with strict=False...")
        try:
            model.load_state_dict(torch.load("final_chess_dl_model.pt", map_location=device), strict=False)
        except Exception as e:
            print(f"Model load failed: {e}")
            
    replay_buffer_states, replay_buffer_policies, replay_buffer_values = [], [], []
    MAX_BUFFER_SIZE = 5000
    
    for iter in range(iterations):
        print(f"\n--- Iteration {iter+1}/{iterations} starts ---")
        all_states, all_policies, all_values = [], [], []
        
        for game in range(games_per_iteration):
            states, policies, values = process_game_vs_bot(model, device, game, simulation=simulation)
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            
        replay_buffer_states.extend(all_states)
        replay_buffer_policies.extend(all_policies)
        replay_buffer_values.extend(all_values)    
        
        if len(replay_buffer_states) > MAX_BUFFER_SIZE:
            replay_buffer_states = replay_buffer_states[-MAX_BUFFER_SIZE:]
            replay_buffer_policies = replay_buffer_policies[-MAX_BUFFER_SIZE:]
            replay_buffer_values = replay_buffer_values[-MAX_BUFFER_SIZE:]
            
        if len(replay_buffer_states) > 0:
            print(f"Training starts (Buffer size: {len(replay_buffer_states)} positions)")
            calculated_loss = train_with_playing_itself_buffer(
                model, optimizer, replay_buffer_states, replay_buffer_policies, replay_buffer_values, device, epochs=3, scaler=scaler
            )
            schedule_tracker.step(calculated_loss)
            
            print("Model updated and saved.")
            torch.save(model.state_dict(), "final_chess_dl_model.pt")
        else:
            print("No data collected from games.")