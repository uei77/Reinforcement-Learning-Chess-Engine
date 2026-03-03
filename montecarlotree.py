import math
import numpy as np

class MCTNode:
    def __init__(self,game,move=None,parent=None,priority=0):
        self.game=game
        self.move=move
        self.parent=parent
        self.children={}
        self.visit=0
        self.value=0
        self.priority = priority
    
    def calculate_value(self):
        if self.visit==0:
            return 0
        else:
            return self.value/self.visit
    
    def selectchild(self,cpuct=1.41): 
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move_uci, child in self.children.items():
            ucb = cpuct * child.priority * (math.sqrt(self.visit) / (1 + child.visit))
            final_score = child.calculate_value() + ucb
            
            if final_score > best_score:
                best_score = final_score  
                best_move = move_uci
                best_child = child
                
        return best_move, best_child
            
            
        