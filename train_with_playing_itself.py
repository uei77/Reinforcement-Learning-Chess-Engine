import torch
from torch.amp import GradScaler, autocast
import os
import chess
from move import Move
from chess_board import board_to_tensor
from monte_carlo_search_tree import monte_carlo_search_algorithm
import copy
import chess.polyglot
import numpy as np
import concurrent.futures

move_record = Move()

def play_move(board, model, device, simulation=100, is_Train=True, move_count=0, opening_book_path=None):
    if opening_book_path and os.path.exists(opening_book_path):
        try:
            with chess.polyglot.open_reader(opening_book_path) as reader:
                entry = reader.choice(board)
                book_move = entry.move
                move_policy = np.zeros(4672)
                idx = move_record._encode_move(book_move)
                if idx is not None:
                    move_policy[idx] = 1.0
                return book_move, move_policy
        except (IndexError, FileNotFoundError):
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

def run_self_play(model, device, simulation_move_limit=100, opening_book_path=None):
    board = chess.Board()
    game_history = []
    while not board.is_game_over() and len(board.move_stack) < 150:
       states = board.copy()      
       move, policy = play_move(board, model, device, simulation=simulation_move_limit, opening_book_path=opening_book_path)  
       game_history.append((states, policy))
       board.push(move) 
    result = get_game_result(board)      
    return prepare_data(game_history, result)

def manager(model, optimizer, device, iterations=10, games_per_iteration=5, simulation=100, opening_book_path=None):
    model.to(device)
    opponent_model = copy.deepcopy(model)
    opponent_model.to(device)
    opponent_model.eval()
    scaler = GradScaler(device.type)
    schedule_tracker = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if os.path.exists("final_chess_rl_model.pt"):
        print("Loading existing model")
        try:
            model.load_state_dict(torch.load("final_chess_rl_model.pt", map_location=device), strict=False)
        except:
            print("Model load failed, starting from scratch.")
    else:
        print("No existing model found. Starting fresh.")
        
    replay_buffer_states = []
    replay_buffer_policies = []
    replay_buffer_values = []
    MAX_BUFFER_SIZE = 5000
    
    for iter in range(iterations):
        print(f"Process {iter+1} starts")
        all_states = []
        all_policies = []
        all_values = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=games_per_iteration) as executor:
            futures = [executor.submit(process_game, model, device, game, simulation, opening_book_path) for game in range(games_per_iteration)]
            for future in concurrent.futures.as_completed(futures):
                states, policies, values = future.result()
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
            print("Training starts")
            calculated_loss = train_with_playing_itself_buffer(
                model, optimizer, replay_buffer_states, replay_buffer_policies, replay_buffer_values, device, epochs=3, scaler=scaler
            )
            schedule_tracker.step(calculated_loss)
            win_rate = evaluate_models(model, opponent_model, device, games=4, simulation=20, opening_book_path=opening_book_path)
            print(f"Win rate of new model: {win_rate:.2f}")
            print("Save the new model")
            torch.save(model.state_dict(), "final_chess_rl_model.pt")
            opponent_model.load_state_dict(model.state_dict())
        else:
            print("No data collected from games")

def model_vs_model(model_white, model_black, device, simulation=20, opening_book_path=None):
    board = chess.Board()
    while not board.is_game_over() and len(board.move_stack) < 150:
        move_count = len(board.move_stack)
        if board.turn == chess.WHITE:
            move = play_move(board, model_white, device, simulation=simulation, is_Train=False, move_count=move_count, opening_book_path=opening_book_path)[0] 
        else:
            move = play_move(board, model_black, device, simulation=simulation, is_Train=False, move_count=move_count, opening_book_path=opening_book_path)[0]
        if move is None:
            break
        board.push(move)
    return get_game_result(board)

def evaluate_models(candidate_model, opponent_model, device, games=10, simulation=20, opening_book_path=None):
    candidate_wins = 0
    opponent_wins = 0
    draws = 0
    
    def play_eval_game(i):
        if i % 2 == 0:
            return model_vs_model(candidate_model, opponent_model, device, simulation, opening_book_path), "candidate_white"
        else:
            return model_vs_model(opponent_model, candidate_model, device, simulation, opening_book_path), "opponent_white"

    with concurrent.futures.ThreadPoolExecutor(max_workers=games) as executor:
        futures = [executor.submit(play_eval_game, i) for i in range(games)]
        for future in concurrent.futures.as_completed(futures):
            result, role = future.result()
            if role == "candidate_white":
                if result == 1.0: candidate_wins += 1
                elif result == -1.0: opponent_wins += 1
                else: draws += 1
            else:
                if result == -1.0: candidate_wins += 1
                elif result == 1.0: opponent_wins += 1
                else: draws += 1
                
    print(f"Results -> Candidate: {candidate_wins} | Best: {opponent_wins} | Draws: {draws}")
    total_score = candidate_wins + 0.5 * draws
    win_rate = total_score / games
    return win_rate

def process_game(model, device, game_idx, simulation=100, opening_book_path=None):
    model.eval() 
    board = chess.Board()
    game_history = []
    with torch.no_grad():
        while not board.is_game_over() and len(board.move_stack) < 150:
            current_move_count = len(board.move_stack)
            with autocast(device_type=device.type, dtype=torch.float16):
                move, policy = play_move(board, model, device, simulation=simulation, move_count=current_move_count, opening_book_path=opening_book_path)  
            if move is None:
                print(f"Warning: Game {game_idx} terminated early (No legal moves found by MCTS).")
                break   
            game_history.append((board.copy(), policy))
            board.push(move)
    final_result = get_game_result(board)
    states, policies, values = prepare_data(game_history, final_result)
    print(f"Game {game_idx+1} is over. Result is {final_result}")
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
        print(f"Train Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} (Value: {loss_value.item():.3f}, Policy: {loss_policy.item():.3f})")
    return loss_sum / epochs