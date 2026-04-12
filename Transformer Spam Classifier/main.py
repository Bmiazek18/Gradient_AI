import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Optional, List, Dict


# --- 1. ARCHITEKTURA MODELU ---

class SelfAttention(nn.Module):
    """
    Implementuje mechanizm Multi-head Self-Attention przy użyciu notacji Einsteina.
    """

    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size musi być podzielny przez heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        N = query.shape[0]
        v_len, k_len, q_len = values.shape[1], keys.shape[1], query.shape[1]

        v = self.values(values).reshape(N, v_len, self.heads, self.head_dim)
        k = self.keys(keys).reshape(N, k_len, self.heads, self.head_dim)
        q = self.queries(query).reshape(N, q_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e9"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, v]).reshape(N, q_len, self.embed_size)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    """
    Blok Transformera: Atencja -> Residual -> Feed Forward -> Residual.
    """

    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        attention = self.attention(values, keys, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Uśrednia wektory słów ignorując tokeny paddingu.
    """
    mask = mask.squeeze(1).squeeze(1)
    x = x * mask.unsqueeze(-1)
    return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)


class SpamClassifier(nn.Module):
    """
    Główny model klasyfikujący Spam/Ham.
    """

    def __init__(self, vocab_size: int, embed_size: int, num_layers: int, heads: int,
                 device: torch.device, forward_expansion: int, dropout: float, max_length: int):
        super(SpamClassifier, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        pooled = masked_mean(out, mask)
        return self.fc_out(pooled)


# --- 2. PRZYGOTOWANIE DANYCH ---

class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: BertTokenizer, max_length: int):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"ham": 0, "spam": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = str(self.df.iloc[index]['text'])
        label = self.label_map[self.df.iloc[index]['label']]

        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(0), torch.tensor(label)


# --- 3. WIZUALIZACJA I EVALUACJA ---

def plot_metrics(history: Dict[str, List[float]]):
    """Generuje wykresy straty i dokładności."""
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss (Trening)', color=color)
    ax1.plot(history['train_loss'], color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy % (Test)', color=color)
    ax2.plot(history['test_acc'], color=color, marker='s', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Postępy treningu modelu Transformer')
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, tokenizer: BertTokenizer):
    """Generuje macierz konfuzji (Heatmap)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            mask = (data != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
            outputs = model(data, mask)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Macierz Konfuzji')
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.show()
    print("\nSzczegółowy Raport Klasyfikacji:")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))


def get_accuracy(loader: DataLoader, model: nn.Module, device: torch.device, tokenizer: BertTokenizer) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            mask = (data != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
            preds = torch.argmax(model(data, mask), dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    model.train()
    return (correct / total) * 100


# --- 4. GŁÓWNY PIPELINE ---

def run_pipeline():
    # Ustawienia
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN, BATCH_SIZE, LR, EPOCHS = 128, 16, 1e-4, 5
    DATA_URL = "hf://datasets/mshenoda/spam-messages/spam_messages_train.csv"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SpamDataset(DATA_URL, tokenizer, MAX_LEN)

    # Podział 80/20
    t_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [t_size, len(dataset) - t_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SpamClassifier(tokenizer.vocab_size, 256, 2, 8, DEVICE, 4, 0.1, MAX_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    print(f"Trening na: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            mask = (data != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data, mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        acc = get_accuracy(test_loader, model, DEVICE, tokenizer)
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(acc)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss / len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    # Wykresy i Raporty
    plot_metrics(history)
    plot_confusion_matrix(model, test_loader, DEVICE, tokenizer)


if __name__ == "__main__":
    run_pipeline()