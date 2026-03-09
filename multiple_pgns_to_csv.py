import os
import chess.pgn
import csv
def convert_all_pgns_to_csv(input_dir, csv_filename, target_total_games=100000):
    pgn_files=[]
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pgn'):
                pgn_files.append(os.path.join(root, file))
    if not pgn_files:
        print(f"Error cannot find pgn files")
        return
    print(f"System found {len(pgn_files)} files ")
    saved_count = 0
    processed_attempts = 0
    with open(csv_filename,'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['moves', 'winner'])
        for pgn_path in pgn_files:
            if saved_count >= target_total_games:
                break
            print(f"Reading {os.path.basename(pgn_path)}")
            with open(pgn_path, encoding="utf-8") as pgn_f:
                while saved_count < target_total_games:
                    game = chess.pgn.read_game(pgn_f)
                    if game is None:
                        print(f"-> {os.path.basename(pgn_path)}")
                        break
                    processed_attempts += 1
                    moves = " ".join([move.uci() for move in game.mainline_moves()])
                    
                    if moves:
                        result = game.headers.get("Result", "*")
                        if result == "1-0":
                            winner = "white"
                        elif result == "0-1":
                            winner = "black"
                        else:
                            winner = "draw"
                        writer.writerow([moves, winner])
                        saved_count += 1
                        
                        if saved_count % 10000 == 0:
                            print(f"{saved_count}/{target_total_games}")
    print("JOB HAS DONE")
if __name__=="__main__":
    INPUT_DIRECTORY = "/kaggle/input" 
    OUTPUT_CSV = "games.csv"
    convert_all_pgns_to_csv(INPUT_DIRECTORY, OUTPUT_CSV, target_total_games=150000)
    