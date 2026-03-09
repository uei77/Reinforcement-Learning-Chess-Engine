import math
import numpy as np
import torch
from torch.amp import autocast 
import chess
from montecarlotree import MCTNode
from chess_board import board_to_tensor
from move import Move

move_control = Move()

def stable(x):
    x = np.array(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=-1e9)
    max_val=np.max(x)
    if np.isinf(max_val):
        return np.zeros_like(x)
    exp = np.exp(x - max_val)
    sum_exp = np.sum(exp)
    if sum_exp == 0:
        return np.zeros_like(x)
        
    return exp / sum_exp

def add_dirichlet_noise(legal_moves, policy_probs, alpha=0.3, epsilon=0.25):
    noise = np.random.dirichlet([alpha] * len(legal_moves))
    for i, move in enumerate(legal_moves):
        policy_probs[move] = (1 - epsilon) * policy_probs[move] + epsilon * noise[i]
    return policy_probs

def get_smart_policy_and_value(board, model, device):
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    model.eval()
    device_type = device.type if hasattr(device, 'type') else ('cuda' if 'cuda' in str(device) else 'cpu')
    
    with torch.no_grad():
        with autocast(device_type=device_type, dtype=torch.float16):
            value_pred, policy_logits = model(tensor)
    
    value = value_pred.item()
    logits = policy_logits.squeeze().cpu().numpy()

    legal_moves = list(board.legal_moves)
    mask = np.full(logits.shape, -1e9, dtype=np.float64) 
    for move in legal_moves:
        idx = move_control._encode_move(move)
        if idx is not None:
            mask[idx] = 0 
    masked_logits = logits + mask
    policy_probs = stable(masked_logits)
    move_probs = {}
    total_prob_check = 0
    for move in legal_moves:
        idx = move_control._encode_move(move)
        if idx is not None:
            prob = policy_probs[idx]
            move_probs[move] = prob
            total_prob_check += prob
            
    if total_prob_check > 0:
        for m in move_probs:
            move_probs[m] /= total_prob_check
            
    return move_probs, value

def run_simulation(root, board, model, device):
    node = root
    search_path = [node]
    current_board = board.copy()
    
    while node.children and not current_board.is_game_over():
        move_uci, node = node.selectchild() 
        if move_uci:
            move = chess.Move.from_uci(move_uci)
            current_board.push(move)
            search_path.append(node)
            
    value = 0
    if not current_board.is_game_over():
        move_probs, value = get_smart_policy_and_value(current_board, model, device)
        for move, prob in move_probs.items():
            if move.uci() not in node.children: 
                node.children[move.uci()] = MCTNode(game=None, move=move, parent=node, priority=prob)
    else:
        result = current_board.result()
        if result == "1-0":
            value = 1.0 if current_board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            value = 1.0 if current_board.turn == chess.BLACK else -1.0
        else:
            value = 0.0
        
    backpropagate(search_path, value, current_board.turn)

def backpropagate(path, value, turn_at_leaf):
    for node in reversed(path):
        value = -value
        node.visit += 1
        node.value += value

def monte_carlo_search_algorithm(board, model, device, numofsimulation=100, is_Train=True,move_count=0):
    root = MCTNode(game=None) 
    move_probs, _ = get_smart_policy_and_value(board, model, device)
    if not move_probs:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, np.zeros(4672)
        prob = 1.0 / len(legal_moves)
        move_probs = {m: prob for m in legal_moves}
        
    if is_Train:
        move_probs = add_dirichlet_noise(list(move_probs.keys()), move_probs)
        
    for move, prob in move_probs.items():
        root.children[move.uci()] = MCTNode(game=None, move=move, parent=root, priority=prob)
        
    for _ in range(numofsimulation):
        run_simulation(root, board, model, device)
        
    moves = []
    visits = []
    for move_uci, child in root.children.items():
        moves.append(move_uci)
        visits.append(child.visit)
        
    visits = np.array(visits)
    if len(visits) == 0 or visits.sum() == 0:
        print("Warning: MCTS failed to decide. Picking random legal move.")
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, np.zeros(4672)
        chosen_move = np.random.choice(legal_moves)
        final_policy = np.zeros(4672)
        idx = move_control._encode_move(chosen_move)
        if idx is not None:
            final_policy[idx] = 1.0
        return chosen_move, final_policy

    if is_Train:
        if move_count < 60:
            prob_distribution = visits / visits.sum()
            chosen_move_uci = np.random.choice(moves, p=prob_distribution)
        else:
            chosen_move_uci = moves[np.argmax(visits)]
    else:
        chosen_move_uci = moves[np.argmax(visits)]
        
    final_policy = np.zeros(4672)
    total_visits = visits.sum()
    
    if total_visits > 0:
        for move_uci, child in root.children.items():
            move_obj = chess.Move.from_uci(move_uci)
            idx = move_control._encode_move(move_obj)
            if idx is not None and idx < 4672:
                final_policy[idx] = child.visit / total_visits
            
    return chess.Move.from_uci(chosen_move_uci), final_policy