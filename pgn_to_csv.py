import chess.pgn
import pandas as pd

def convert_pgn_to_csv_fixed_amount(pgn_filename, csv_filename, target_game_count=300000):
    games_data = []
    saved_count = 0
    processed_attempts = 0
    
    print(f"Reading {pgn_filename} until {target_game_count} valid games are found...")
    
    with open(pgn_filename, encoding="utf-8") as pgn_file:
        while saved_count < target_game_count:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                print("End of PGN file reached before hitting the target.")
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
                
                games_data.append({
                    'moves': moves,
                    'winner': winner
                })
                saved_count += 1
            if saved_count % 1000 == 0 and moves:
                print(f"Progress: {saved_count}/{target_game_count} games saved... (Checked {processed_attempts} items)")
    df = pd.DataFrame(games_data)
    df.to_csv(csv_filename, index=False)
    print(f"\nSuccess! Exactly {len(df)} valid games saved to {csv_filename}")

if __name__ == "__main__":
    INPUT_PGN = "lichess_elite_2020-08.pgn"
    OUTPUT_CSV = "games.csv"
    
    convert_pgn_to_csv_fixed_amount(INPUT_PGN, OUTPUT_CSV, target_game_count=500000)