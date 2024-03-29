import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif isinstance(model, nn.LSTM):
        for name, param in model.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output = [batch size, seq len, hidden dim * num directions]
        attn_weights = self.attn(lstm_output).squeeze(2)
        # attn_weights = [batch size, seq len]

        softmax_weights = F.softmax(attn_weights, dim=1)
        # softmax_weights = [batch size, seq len]

        # Reshape weights to [batch size, 1, seq len]
        softmax_weights = softmax_weights.unsqueeze(1)

        # Apply attention weights to lstm_output
        weighted_output = torch.bmm(softmax_weights, lstm_output).squeeze(1)
        # weighted_output = [batch size, hidden dim * num directions]

        # Return both output and weights
        return weighted_output, softmax_weights.squeeze(1)


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(_init_weights)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            # Concatenate the hidden states from the last layer of the LSTM for both directions
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            # Use the hidden state from the last layer of the LSTM
            hidden = self.dropout(hidden[-1])

        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]

        return prediction


class LSTMWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        # Set to True if you want to fine-tune embeddings
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(_init_weights)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output = [batch size, seq len, hidden dim * num directions]

        attn_output, attn_weights = self.attention(lstm_output)
        # attn_output = [batch size, hidden dim * num directions]

        prediction = self.fc(self.dropout(attn_output))
        # prediction = [batch size, output dim]

        return prediction, attn_weights


class BERTClassifier(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, tweet_token_ids, attention_mask=None):
        output = self.transformer(
            tweet_token_ids, attention_mask=attention_mask, output_attentions=True
        )
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction
