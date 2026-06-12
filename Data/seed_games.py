"""
This script is to load all games from pgn file
For this project I used 2026-05 games from lichess, ~90M games
"""
import zstandard as zstd
import io
import chess.pgn
import os
from dotenv import load_dotenv

load_dotenv()

lichess_db = os.getenv("lichess_db")

if lichess_db is None:
    raise ValueError("Missing 'lichess_db' in .env file.")

# only fetching 100,000/90,000,000 games to train
number_of_games = 100000

# decompress loop, lichess games are stored as .pgn.zst
with open(lichess_db, "rb") as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        # loop to iterate through games
        for _ in range(number_of_games):
            game = chess.pgn.read_game(text_stream)

            if game is None:
                break

            board = game.board()
            # loop to iterate through moves
            for move in game.mainline_moves():
                board.push(move)




