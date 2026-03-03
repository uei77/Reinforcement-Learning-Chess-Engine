import chess
class Move():
    def __init__(self):
        self.move_to_idx={}
        self.idx_to_move={}
        self._initialize_moves()
                
    def _initialize_moves(self):
        move_idx=0
        for from_square in range(64):
            for to_square in range(64):
                if from_square == to_square:
                    continue
                move_uci=chess.SQUARE_NAMES[from_square]+chess.SQUARE_NAMES[to_square]
                
                from_rank=chess.square_rank(from_square)
                to_rank=chess.square_rank(to_square)
                file_diff = abs(chess.square_file(from_square) - chess.square_file(to_square))
                
                if file_diff <= 1 and ((from_rank==6 and to_rank==7) or (from_rank==1 and to_rank==0)):
                    self._add_move(move_uci, move_idx)
                    move_idx += 1
                    for promotion in ['q','r','b','n']:
                        promotion_move = move_uci + promotion
                        self._add_move(promotion_move, move_idx)
                        move_idx += 1
                else:
                    self._add_move(move_uci, move_idx)
                    move_idx += 1
                
    def _add_move(self, move_uci, move_idx):
        self.move_to_idx[move_uci] = move_idx
        self.idx_to_move[move_idx] = move_uci
        
    def _encode_move(self, move: chess.Move):
        return self.move_to_idx.get(move.uci(), None)
        
    def _decode_move(self, move_idx):
        uci = self.idx_to_move.get(move_idx, None)      
        if uci:
            return chess.Move.from_uci(uci)
        return None
                