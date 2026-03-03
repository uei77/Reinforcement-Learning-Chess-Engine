import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chess
import uvicorn
from uci_decoder import load_model, find_best_move
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ChessEngineAPI")
models={}
MODEL_PATH = "final_chess_dl_model.pt"
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Chess Engine ")
    try:
        model,device=load_model(MODEL_PATH)
        models["chess_model"]=model
        models["device"]=device
        logger.info(f"Model succesfully load on {device}")
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        raise e
    yield
    logger.info("Shutting down, clearing memory...")
    models.clear()
app = FastAPI(
    title="1000 - 1100 ELO CHESS ENGINE",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
class EngineRequest(BaseModel):
    fen: str = Field(...)
class EngineResponse(BaseModel):
    best_move: str = Field(...)
class HealthResponse(BaseModel):
    status: str
    device: str
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    device_info = str(models.get("device", "Unknown"))
    return HealthResponse(status="API and Model Active", device=device_info)
@app.post("/api/v1/engine/move", response_model=EngineResponse, tags=["Game"])
async def get_engine_move(request:EngineRequest):
    logger.info(f"Received FEN: {request.fen}")
    try:
        board=chess.Board(request.fen)
    except ValueError :
        logger.warning(f"Invalid FEN received: {request.fen}")
        raise HTTPException(status_code=400, detail="Invalid FEN string provided.")
    if board.is_game_over():
        logger.info("Game is already over.")
        raise HTTPException(status_code=400, detail="Game is already over.")
    try:
        model=models.get("chess_model")
        device=models.get("device")
        if not model or not device:
            raise ValueError("Model not loaded into memory.")
        best_move_uci = find_best_move(board, model, device)
        logger.info(f"Calculated best move: {best_move_uci}")
        return EngineResponse(best_move=best_move_uci)
    except Exception as e:
        logger.error(f"Engine error during move calculation: {e}")
        raise HTTPException(status_code=500, detail="Chess engine encountered an error calculating the move.")
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False
    )
    