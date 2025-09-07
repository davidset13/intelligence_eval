import pandas as pd
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

ml_dataset = pd.read_csv(os.path.join(os.getcwd(), "utility", "code_detection_dataset.csv"))

@dataclass
class Row:
    text: str
    start_index: int
    end_index: int

class SpanWindowDataset(Dataset):

    def __init__(self, rows: list[Row], max_len: int = 2048):
        self.rows = rows
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.rows)

    def __print__(self) -> str:
        return f"""SpanWindowDataset(
            rows={self.rows[0:5]},
            ...
        )"""
    
    @staticmethod
    def _to_bytes(s: str) -> list[int]:
        return list(s.encode("utf-8", errors="replace"))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        text, start, end = self._to_bytes(row.text), row.start_index, row.end_index

        return torch.tensor(text, dtype=torch.long), torch.tensor(start), torch.tensor(end), torch.tensor(len(text))


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    seqs, starts, ends, lengths = zip(*batch)
    maxTokens = max(int(L) for L in lengths)
    B = len(seqs)
    out = torch.full((B, maxTokens), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxTokens), dtype=torch.bool)
    for i, (seq, L) in enumerate(zip(seqs, lengths)):
        out[i, :L] = seq
        attn[i, :L] = True
    s_pos = torch.stack(starts)
    e_pos = torch.stack(ends)
    return out, attn, s_pos, e_pos


class CodeMLModel(nn.Module):

    def __init__(self, vocab_size=256, d_model=256, lstm_hidden=256, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model, padding_idx = 0)
        self.lstm: nn.LSTM = nn.LSTM(d_model, lstm_hidden, num_layers = lstm_layers, batch_first = True, bidirectional = True, dropout = dropout)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.start_head: nn.Linear = nn.Linear(lstm_hidden * 2, 1)
        self.end_head: nn.Linear = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, X: torch.Tensor, attn_mask: torch.Tensor):
        X = X.contiguous()
        self.lstm.flatten_parameters()
        emb = self.embedding(X).contiguous()
        packed_out, _ = self.lstm(emb)
        h: torch.Tensor = self.dropout(packed_out)
        start_logits: torch.Tensor = self.start_head(h).squeeze(-1)
        end_logits: torch.Tensor = self.end_head(h).squeeze(-1)
        
        start_logits = start_logits.masked_fill(~attn_mask, -1e9)
        end_logits = end_logits.masked_fill(~attn_mask, -1e9)
        return start_logits, end_logits


def span_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor, end_positions: torch.Tensor):
    
    B, T = start_logits.shape
    loss_f = nn.CrossEntropyLoss(ignore_index = -100)
    loss_s = loss_f(start_logits, start_positions)
    loss_e = loss_f(end_logits, end_positions)
    return (loss_s + loss_e) / 2


def train_model(train_rows: list[Row], val_rows: list[Row], max_len=2048, epochs=3, device="cuda"):
    
    train_ds = SpanWindowDataset(train_rows, max_len=max_len)
    val_ds   = SpanWindowDataset(val_rows, max_len=max_len)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_pad, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_pad, num_workers=2)

    model = CodeMLModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, attn, ys, ye in train_dl:
            x, attn, ys, ye = x.to(device), attn.to(device), ys.to(device), ye.to(device)
            opt.zero_grad()
            start_logits, end_logits = model(x, attn)
            loss = span_loss(start_logits, end_logits, ys, ye)
            loss.backward()
            opt.step()
            total += float(loss)
        
        avg = total / len(train_dl)
        model.eval()

        with torch.no_grad():
            em, iou, n = 0.0, 0.0, 0.0
            for x, attn, ys, ye in val_dl:
                x, attn, ys, ye = x.to(device), attn.to(device), ys.to(device), ye.to(device)
                s_logits, e_logits = model(x, attn)
                ps = s_logits.argmax(dim=-1)
                pe = e_logits.argmax(dim=-1)

                mask = (ys != -100) & (ye != -100)
                if mask.any():
                    eq = ((ps == ys) & (pe == ye) & mask).sum().item()
                    em += eq

                    for i in torch.nonzero(mask, as_tuple=False).squeeze(1).tolist():
                        s1, e1 = int(ps[i].item()), int(pe[i].item())
                        s2, e2 = int(ys[i].item()), int(ye[i].item())
                        inter = max(0, min(e1, e2) - max(s1, s2) + 1)
                        union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
                        iou += 0.0 if union <= 0 else inter / union
                        n += 1
                    
                    em_rate = em / max(1, len(val_ds))
                    mean_iou = iou / max(1, n)
        
        print(f"Epoch {epoch}: train_loss={avg:.4f}  val_EM={em_rate:.3f}  val_IoU={mean_iou:.3f}")

    return model

def main():
    samples = [Row(row.text, row.start_index, row.end_index) for row in ml_dataset.itertuples(index=False)] # type: ignore
    
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]

    model = train_model(train_samples, val_samples, max_len=2048, epochs=3, device="cpu")

if __name__ == "__main__":
    main()
