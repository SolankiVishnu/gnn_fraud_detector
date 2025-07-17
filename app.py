import streamlit as st
import torch
import pickle
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# GCN Model Definition
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

@st.cache_resource
def load_model_and_scaler():
    model = GCN(in_channels=16, hidden_channels=64, out_channels=2)
    model.load_state_dict(torch.load("gnn_model.pt", map_location=torch.device('cpu')))
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("💳 GNN-based Fraud Detection")

st.subheader("Enter transaction features:")
fields = [
    "Transaction_Amount", "Account_Balance", "Device_Type", "Location", "Merchant_Category",
    "IP_Address_Flag", "Previous_Fraudulent_Activity", "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d", "Card_Type", "Card_Age",
    "Transaction_Distance", "Authentication_Method", "Risk_Score", "Is_Weekend"
]

user_input = []
for field in fields:
    if "Amount" in field or "Balance" in field or "Distance" in field or "Score" in field:
        val = st.number_input(field, value=1000.0)
    else:
        val = st.number_input(field, value=1)
    user_input.append(val)

if st.button("Predict Fraud"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled = scaler.transform(input_array)
    x = torch.tensor(scaled, dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # dummy edge
    data = Data(x=x, edge_index=edge_index)
    with torch.no_grad():
        output = model(data)
        prediction = torch.argmax(output, dim=1).item()
    if prediction == 1:
        st.error("⚠️ FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("✅ Legitimate Transaction")
