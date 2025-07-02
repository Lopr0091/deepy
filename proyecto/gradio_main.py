import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modelos por tipo ===

# --- Modelo por caracteres (delamancha / delamanchanuevo) ---
class ModeloChar(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, estados=None):
        emb = self.embedding(x)
        salida, estados = self.lstm(emb, estados)
        logits = self.fc(salida)
        return logits, estados

# --- Modelo por palabras ---
class ModeloPalabras(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            dropout=0.3, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout_fc = nn.Dropout(0.3)

    def forward(self, x, estados=None):
        x = self.dropout(self.layernorm(self.embedding(x)))
        salida, estados = self.lstm(x, estados)
        salida = self.dropout_fc(self.fc(salida))
        return salida, estados

# --- Modelo de sentimientos (transformer) ---
class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x).permute(1, 0, 2)
        out = self.transformer_encoder(emb).mean(dim=0)
        return torch.sigmoid(self.fc(self.dropout(out))).squeeze()

# === Utilidades por modelo ===

def generar_texto_palabras(frase, modelo, token_to_id, id_to_token, seq_length=10, top_k=10, temperatura=1.2):
    entrada = frase.lower().split()
    entrada_ids = [token_to_id.get(w, 0) for w in entrada][-seq_length:]
    while len(entrada_ids) < seq_length:
        entrada_ids.insert(0, 0)
    input_tensor = torch.tensor([entrada_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        salida, _ = modelo(input_tensor)
        logits = salida[:, -1, :] / temperatura
        probas = F.softmax(logits, dim=-1).squeeze()
        top_probs, top_indices = torch.topk(probas, top_k)
        sugerencias = [id_to_token[idx.item()] for idx in top_indices[:3]]
    return " | ".join(sugerencias)

def clasificar_sentimiento(texto, modelo, vocab, tokenizer, max_len=200):
    tokens = vocab(tokenizer(texto.lower()))[:max_len]
    if not tokens:
        return "Texto vacío o inválido"
    padded = tokens + [vocab["<pad>"]] * (max_len - len(tokens))
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = modelo(input_tensor)
    return "Negativo" if pred.item() > 0.5 else "Positivo"

# === Cargar modelos ===

modelos = {}

# --- Palabras ---
tokens_path = "data/texto_limpio_palabras.txt"
with open(tokens_path, "r", encoding="utf-8") as f:
    tokens = f.read().split()
vocab_pal = sorted(set(tokens))
token_to_id_pal = {w: i for i, w in enumerate(vocab_pal)}
id_to_token_pal = {i: w for w, i in token_to_id_pal.items()}
modelo_pal = ModeloPalabras(len(vocab_pal)).to(device)
modelo_pal.load_state_dict(torch.load("modelos/mejor_modelo_lstm_palabras.pth", map_location=device))
modelo_pal.eval()
modelos["palabras"] = modelo_pal

# --- Sentimientos ---
tokenizer_sent = get_tokenizer("basic_english")
train_iter = IMDB(split='train')
vocab_sent = build_vocab_from_iterator((tokenizer_sent(text) for _, text in train_iter),
                                       specials=["<pad>", "<unk>"])
vocab_sent.set_default_index(vocab_sent["<unk>"])
modelo_sent = SentimentTransformer(len(vocab_sent)).to(device)
modelo_sent.load_state_dict(torch.load("modelos/modelo_sentimiento.pth", map_location=device))
modelo_sent.eval()
modelos["sentimientos"] = modelo_sent

# === Interfaz ===

def interfaz(texto, tipo):
    if tipo == "palabras":
        return generar_texto_palabras(texto, modelos["palabras"], token_to_id_pal, id_to_token_pal)
    elif tipo == "sentimientos":
        return clasificar_sentimiento(texto, modelos["sentimientos"], vocab_sent, tokenizer_sent)
    else:
        return "Modelo no implementado aún."

demo = gr.Interface(
    fn=interfaz,
    inputs=[
        gr.Textbox(label="Texto de entrada"),
        gr.Radio(["palabras", "sentimientos"], label="Modelo a usar")
    ],
    outputs="text",
    title="Selector de Modelo NLP - Quijote y Sentimientos",
    description="Elige entre generación de texto estilo Quijote o clasificación de sentimiento con IMDB."
)

if __name__ == "__main__":
    demo.launch()
