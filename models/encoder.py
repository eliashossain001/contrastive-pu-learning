import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        attn_out, _ = self.attn(x, x, x)
        return attn_out.permute(1, 0, 2).mean(dim=1)  # (batch_size, hidden_dim)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        h, _ = self.lstm(x)
        attn_out = self.attn(h)
        return self.fc(attn_out)
