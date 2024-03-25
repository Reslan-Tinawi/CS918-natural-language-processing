import torch
import torch.nn as nn


class LSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_dim,
        output_dim,
        num_layers,
        bidirectional,
        dropout,
        padding_idx,
    ):
        super(LSTMClassifier, self).__init__()

        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_matrix, padding_idx=padding_idx
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_matrix.size(1),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


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
        # nn.LSTM(embedding_dim, 256, batch_first=True)
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
