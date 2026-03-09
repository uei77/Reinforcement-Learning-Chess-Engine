import os
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import chess
import chess.engine
import numpy as np
import random
from neural_network import chess_neural_network
from move import Move
from train_with_playing_itself import play_move, prepare_data, train_with_playing_itself_buffer, get_game_result

move_record = Move()
STOCKFISH_CMD = "stockfish"
TARGET_SKILL_LEVEL = 5
GAMES_TO_PLAY = 10
MCTS_SIMULATIONS = 800
LEARNING_RATE = 0.001
FINAL_MODEL_NAME = "final_chess_rl_model.pt"
OPENING_BOOK_PATH = "Perfect2021.bin"

def process_game_vs_stockfish(model, device, engine, game_idx, simulation):
    model.eval()
    board = chess.Board()
    game_history = []
    
    model_is_white = random.choice([True, False])
    print(f"Game {game_idx+1} starts. RL Model is {'WHITE' if model_is_white else 'BLACK'}")
    
    with torch.no_grad():
        while not board.is_game_over() and len(board.move_stack) < 400:
            current_move_count = len(board.move_stack)
            
            if (board.turn == chess.WHITE and model_is_white) or (board.turn == chess.BLACK and not model_is_white):
                with autocast(device_type=device.type, dtype=torch.float16):
                    move, policy = play_move(board, model, device, simulation=simulation, move_count=current_move_count, opening_book_path=OPENING_BOOK_PATH)
                if move is None:
                    break
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = chess_neural_network().to(device)
    
    if os.path.exists(FINAL_MODEL_NAME):
        try:
            model.load_state_dict(torch.load(FINAL_MODEL_NAME, map_location=device), strict=False)
        except Exception:
            pass
            
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler(device.type)
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_CMD)
        engine.configure({"Skill Level": TARGET_SKILL_LEVEL})
    except Exception as e:
        print(f"Stockfish error: {e}")
        return

    all_states, all_policies, all_values = [], [], []
    
    for game in range(GAMES_TO_PLAY):
        states, policies, values = process_game_vs_stockfish(model, device, engine, game, MCTS_SIMULATIONS)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        
    engine.quit()
    
    if len(all_states) > 0:
        train_with_playing_itself_buffer(
            model, optimizer, all_states, all_policies, all_values, device, epochs=3, scaler=scaler
        )
        torch.save(model.state_dict(), FINAL_MODEL_NAME)
        print("Model updated from Stockfish matches and saved.")
        
if __name__ == "__main__":
    main()