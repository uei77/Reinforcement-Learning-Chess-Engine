import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import chess
import uvicorn
from uci_decoder import load_model, find_best_move
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from cachetools import TTLCache  
from functools import lru_cache
import uuid  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ChessEngineAPI")

MODEL_PATH = os.getenv("MODEL_PATH", "final_chess_rl_model.pt")
API_KEY = os.getenv("API_KEY", "default_key_change_this")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1,null").split(",")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

models = {}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

move_cache = TTLCache(maxsize=100, ttl=300)
games = {} 
total_engine_wins = 0.0
total_player_wins = 0.0
total_draws = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Chess Engine")
    try:
        model, device = load_model(MODEL_PATH)
        models["chess_model"] = model
        models["device"] = device
        logger.info(f"Model successfully loaded on {device}")
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        raise e
    yield
    logger.info("Shutting down, clearing memory...")
    models.clear()
    executor.shutdown(wait=True)

app = FastAPI(
    title="1900 - 2000 ELO Chess Engine",
    version="1.2.0", 
    lifespan=lifespan
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=False,    
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class EngineRequest(BaseModel):
    fen: str = Field(..., min_length=1, max_length=100)
    game_id: str = Field(None, description="Optional game ID for tracking sessions")
    num_simulations: int = Field(800, ge=1, le=800, description="Number of MCTS simulations (1-800, default 800)")

    @validator('fen')
    def validate_fen(cls, v):
        if not all(c.isalnum() or c in ' /-' for c in v):
            raise ValueError('Invalid characters in FEN')
        return v

class EngineResponse(BaseModel):
    best_move: str = Field(...)
    game_over: bool = Field(False)
    result: str = Field(None, description="Game result if over, e.g., '1-0', '0-1', '1/2-1/2'")

class HealthResponse(BaseModel):
    status: str
    device: str

class CreateGameResponse(BaseModel):
    game_id: str

class ResultsResponse(BaseModel):
    engine_wins: float
    player_wins: float
    draws: float
    total_games: int

def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Verify API key from header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the chess UI (eliminates CORS by making page same-origin as API)."""
    return FileResponse("index.html")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    device_info = str(models.get("device", "Unknown"))
    return HealthResponse(status="API and Model Active", device=device_info)

@app.post("/api/v1/game/create", response_model=CreateGameResponse, tags=["Game"], dependencies=[Depends(verify_api_key)])
async def create_game() -> CreateGameResponse:
    """Create a new game session and return its unique ID."""
    game_id = str(uuid.uuid4())
    games[game_id] = {"history": [], "fen": chess.STARTING_FEN, "result": None}
    logger.info(f"New game created: {game_id}")
    return CreateGameResponse(game_id=game_id)


@app.get("/api/v1/game/results", response_model=ResultsResponse, tags=["Game"], dependencies=[Depends(verify_api_key)])
async def get_results() -> ResultsResponse:
    """Get total results across all games."""
    total_games = len([g for g in games.values() if g["result"] is not None])
    return ResultsResponse(
        engine_wins=total_engine_wins,
        player_wins=total_player_wins,
        draws=total_draws,
        total_games=total_games
    )

@app.post("/api/v1/engine/move", response_model=EngineResponse, tags=["Game"], dependencies=[Depends(verify_api_key)])
async def get_engine_move(request: EngineRequest) -> EngineResponse:
    """Compute best move for given FEN using cached MCTS. Updates game state if game_id provided."""
    logger.info(f"Received FEN: {request.fen}, Game ID: {request.game_id}")
    try:
        board = chess.Board(request.fen)
    except ValueError:
        logger.warning(f"Invalid FEN received: {request.fen}")
        raise HTTPException(status_code=400, detail="Invalid FEN string provided.")
    
    game_over = board.is_game_over()
    result = None
    if game_over:
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"  
        else:
            result = "1/2-1/2"
        logger.info("Game is already over.")
       
        if request.game_id and request.game_id in games:
            games[request.game_id]["result"] = 1 if result == "1-0" else 0 if result == "0-1" else 0.5
            global total_engine_wins, total_player_wins, total_draws
            if result == "1-0":
                total_engine_wins += 1
            elif result == "0-1":
                total_player_wins += 1
            else:
                total_draws += 0.5
    
    if game_over:
        return EngineResponse(best_move="", game_over=True, result=result)
    
    if request.game_id:
        if request.game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found.")
        games[request.game_id]["fen"] = request.fen
    
    cache_key = f"{request.fen}_{request.num_simulations}"  
    if cache_key in move_cache:
        logger.info("Returning cached move")
        best_move_uci = move_cache[cache_key]
    else:
        try:
            model = models.get("chess_model")
            device = models.get("device")
            if not model or not device:
                raise ValueError("Model not loaded into memory.")
            
            loop = asyncio.get_event_loop()
            best_move_uci = await loop.run_in_executor(executor, find_best_move, board, model, device, request.num_simulations)
            
            move_cache[cache_key] = best_move_uci
            logger.info(f"Calculated and cached best move: {best_move_uci}")
        except Exception as e:
            logger.error(f"Engine error during move calculation: {e}", exc_info=True)  
            raise HTTPException(status_code=500, detail="Chess engine encountered an error calculating the move.")
    
    if request.game_id:
        games[request.game_id]["history"].append(best_move_uci)
    
    return EngineResponse(best_move=best_move_uci, game_over=False, result=None)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False
    )