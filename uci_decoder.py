import sys
import requests
import chess
import chess.polyglot
import torch
import traceback
from neural_network import chess_neural_network
from monte_carlo_search_tree import monte_carlo_search_algorithm

def load_model(model_path: str) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = chess_neural_network().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        sys.stderr.write(f"Model uploaded ({device}).\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"Could not load model: {e}\n")
        sys.stderr.flush()
        sys.exit(1)
    return model, device

def update_board(board: chess.Board, tokens: list[str]) -> None:
    if "startpos" in tokens:
        board.set_fen(chess.STARTING_FEN)
        if "moves" in tokens:
            moves_start_idx = tokens.index("moves") + 1
            for move_uci in tokens[moves_start_idx:]:
                board.push_uci(move_uci)
    elif "fen" in tokens:
        fen_start_idx = tokens.index("fen") + 1
        fen_string = " ".join(tokens[fen_start_idx:fen_start_idx + 6])
        board.set_fen(fen_string)
        if "moves" in tokens:
            moves_start_idx = tokens.index("moves") + 1
            for move_uci in tokens[moves_start_idx:]:
                board.push_uci(move_uci)

def get_book_move(board: chess.Board, book_path: str = "Perfect2021.bin") -> str:
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            entry = reader.weighted_choice(board)
            return entry.move.uci()
    except:
        return None

def get_syzygy_move_online(board: chess.Board) -> str:
    piece_count = len(board.piece_map())
    if piece_count > 7:
        return None
        
    try:
        fen = board.fen()
        url = f"http://tablebase.lichess.ovh/standard?fen={fen}"
        response = requests.get(url, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            if "moves" in data and len(data["moves"]) > 0:
                best_move = data["moves"][0]["uci"]
                return best_move
        return None
    except Exception:
        return None

def find_best_move(board: chess.Board, model: torch.nn.Module, device: torch.device) -> str:
    book_move = get_book_move(board)
    if book_move:
        return book_move
        
    syzygy_move = get_syzygy_move_online(board)
    if syzygy_move:
        return syzygy_move

    try:
        move, _ = monte_carlo_search_algorithm(
            board=board,
            model=model,
            device=device,
            numofsimulation=400, 
            is_Train=False, 
            move_count=len(board.move_stack)
        )
        if move is None:
            sys.stderr.write("Cannot find MCTS move playing random move\n")
            move = list(board.legal_moves)[0]
        return move.uci()
    except Exception as e:
        sys.stderr.write(f"Engine error : {e}\n")
        sys.stderr.write(traceback.format_exc())
        return list(board.legal_moves)[0].uci()

def uci_play():
    model_path = "final_chess_dl_model.pt"
    model, device = load_model(model_path)
    board = chess.Board()
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            tokens = line.split()
            command = tokens[0]

            if command == "uci":
                print("id name engine1")
                print("id author ab104")
                print("uciok")
                sys.stdout.flush()
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
            elif command == "ucinewgame":
                board.reset()
            elif command == "position":
                update_board(board, tokens)
            elif command == "go":
                best_move_uci = find_best_move(board, model, device)
                print(f"bestmove {best_move_uci}")
                sys.stdout.flush()
            elif command == "quit":
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            sys.stderr.write(f"UCI loop error: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    uci_play()