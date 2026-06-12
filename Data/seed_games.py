"""
This script is to load all games from pgn file
For this project I used 2026-05 games from lichess, ~90M games
"""
import zstandard as zstd
import io
import chess.pgn
import os
from dotenv import load_dotenv
import torch

load_dotenv()

lichess_db = os.getenv("lichess_db")

if lichess_db is None:
    raise ValueError("Missing 'lichess_db' in .env file.")

# only fetching 100,000/90,000,000 games to train
number_of_games = 100000
os.makedirs("data/tensors", exist_ok=True)

buffer = []
chunk_counter = 1

# function to save chess positions to torch tensors
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    # torch binary tensor (12 pieces, 8 x 8 board)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    
    # translating piece to int representations
    piece_mapping = {
        (chess.PAWN, True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3,
        (chess.QUEEN, True): 4,
        (chess.KING, True): 5,
        (chess.PAWN, False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9,
        (chess.QUEEN, False): 10,
        (chess.KING, False): 11
    }
    # loop to iterate through entire chess board for positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            row = chess.square_rank(square)
            col = chess.square_file(square)

            piece_type = piece.piece_type
            is_white = piece.color == chess.WHITE
            channel = piece_mapping[(piece_type, is_white)]
            tensor[channel, row, col] = 1.0

    return tensor

print(f"Saving games to tensors...")

# decompress loop, lichess games are stored as .pgn.zst
with open(lichess_db, "rb") as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        # loop to iterate through games
        for i in range(number_of_games):
            game = chess.pgn.read_game(text_stream)

            if game is None:
                break

            board = game.board()

            # loop to iterate through moves
            for node in game.mainline():
                board.push(node.move)
                eval_pov = node.eval()

                if len(buffer) == 10000:
                    # clears ram buffer by saving tensor then clearing
                    x_list = [b[0] for b in buffer]
                    y_list = [b[1] for b in buffer]

                    x_batch = torch.stack(x_list)
                    y_batch = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
                    torch.save({"X": x_batch, "Y": y_batch}, f"data/tensors/chunk_{chunk_counter:03d}.pt")
                    buffer = []
                    chunk_counter += 1

                if eval_pov is not None:
                    y_val = eval_pov.white().score(mate_score=10000)
                    x_tensor = board_to_tensor(board)
                    buffer.append((x_tensor, y_val))

            if i % 5000 == 0:
                print(f"{i} games saved.")


# final buffer clearance because early loop only saves every 10,000 games           
if len(buffer) > 0:
    x_list = [b[0] for b in buffer]
    y_list = [b[1] for b in buffer]

    x_batch = torch.stack(x_list)
    y_batch = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    torch.save({"X": x_batch, "Y": y_batch}, f"data/tensors/chunk_{chunk_counter:03d}.pt")
    buffer = []

print(f"Game extraction complete. Saved {chunk_counter} chunks.")

                
            







