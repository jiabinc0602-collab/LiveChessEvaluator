"""
Export the trained chess evaluator model to ONNX format.
"""
import torch
from model.chess_model import ChessEvaluator

# hyper params
d_model = 256
num_heads = 8
num_layers = 6
batch_size = 2048
lr = 3e-4

model = ChessEvaluator(d_model, num_heads, num_layers)

state_dict = torch.load('chess_evaluator.pth', map_location='cpu')

model.load_state_dict(state_dict)

model.eval()

dummy = torch.randn(1, 12, 8, 8)

torch.onnx.export(model, dummy, "chess_evaluator.onnx", export_params=True, opset_version=18)
