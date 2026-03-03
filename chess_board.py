import chess
import numpy as np
import torch
def board_to_tensor(board:chess.Board):
    
    square_status=18#wp,wk,wb,wr,wq,wking,bp,bk,bb,br,bq,bking,null,enpassant,castling
    squares=np.zeros((8,8,square_status),dtype='float32')#8x8 board 13 square status
    fill_layers(board,squares)
    return torch.from_numpy(squares).permute(2,0,1)
def fill_layers(board: chess.Board, squares: np.ndarray):
    pieces=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
    colours=[chess.WHITE,chess.BLACK]
    for c in colours:
        for piece_idx ,piece_type in enumerate(pieces):
            square_layer=piece_idx + (6 if c==chess.BLACK else 0)
            for square in board.pieces(piece_type,c):
                row=chess.square_rank(square)
                column=chess.square_file(square)
                squares[row,column,square_layer]=1.0
    if board.turn == chess.WHITE:
        squares[:, :, 12] = 1.0
    else:
        squares[:, :, 12] = 0.0
    if board.ep_square is not None:
        ep_row=chess.square_rank(board.ep_square)   
        ep_column=chess.square_file(board.ep_square)     
        squares[ep_row, ep_column, 13] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        squares[:, :, 14] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        squares[:, :, 15] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        squares[:, :, 16] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        squares[:, :, 17] = 1.0

if __name__ == "__main__":
    board_start = chess.Board()
    board_tensor = board_to_tensor(board_start)    