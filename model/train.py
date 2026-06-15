"""
This script is used to train the model
"""
import torch
from torch import nn
import pathlib
from torch.utils.data import Dataset, DataLoader
from chess_model import ChessEvaluator

class ChessDataset(Dataset):
    def __init__(self):
        super().__init__()
        tensor_path = pathlib.Path("data/tensors").glob("*.pt")
        all_X = []
        all_Y = []
        # appending all X and Y values from tensors
        for tensor in tensor_path:
            data = torch.load(tensor, map_location='cpu')
            all_X.append(data["X"])
            all_Y.append(data["Y"])
        self.X = torch.cat(all_X, dim = 0)
        raw_Y = torch.cat(all_Y, dim = 0)

        # sigmoid temperature scale to noramlize probabilities
        self.Y = 1.0 / (1.0 + torch.exp(-raw_Y/400.0))


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    

if __name__ == '__main__':
    # hyper params
    d_model = 256
    num_heads = 8
    num_layers = 6
    batch_size = 1024
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(
        dataset=ChessDataset(),
        batch_size=batch_size,
        shuffle=True
    )

    model = ChessEvaluator(d_model=d_model, 
                           num_heads=num_heads, 
                           num_layers=num_layers
                           ).to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    criterion= nn.BCELoss()

    epochs = 10
    
    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            # move data to GPU/CPU
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            # forward
            predictions = model(X_batch)

            # loss
            loss = criterion(predictions, Y_batch)

            # back prop
            loss.backward()

            # update weights
            optimizer.step()

            running_loss += loss.item()

            if batch_idx > 0 and batch_idx % 1000 == 0:
                avg_loss = running_loss / 1000
                print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx} | Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        print(f"--- Completed Epoch {epoch+1} ---")
        
    print("Training complete. Saving weights...")
    torch.save(model.state_dict(), "chess_evaluator.pth")
    print("Saved to chess_evaluator.pth")








