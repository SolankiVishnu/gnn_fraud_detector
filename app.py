from flask import Flask, render_template, request
import torch
import numpy as np
import pickle
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

app = Flask(__name__)

# Load model and scaler
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(16, 64, 2)
model.load_state_dict(torch.load("gnn_model.pt", map_location=torch.device('cpu')))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [float(request.form[f]) for f in request.form]
        scaled = scaler.transform([features])
        x = torch.tensor(scaled, dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            output = model(data)
            pred = torch.argmax(output, dim=1).item()
            prediction = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
