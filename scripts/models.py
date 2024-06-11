import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_weights(model):
    """
    Initializes the weights of the given model.

    Args:
        model (nn.Module): The model for which to initialize the weights.

    Returns:
        None
    """
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
    """
    Attention module that calculates attention weights and applies them to the LSTM output.

    Args:
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        attn (nn.Linear): Linear layer to calculate attention weights.

    Methods:
        forward(lstm_output): Performs the forward pass of the attention module.

    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Performs the forward pass of the attention module.

        Args:
            lstm_output (torch.Tensor): The output of the LSTM layer.
                Shape: [batch size, seq len, hidden dim * num directions]

        Returns:
            weighted_output (torch.Tensor): The output after applying attention weights to the LSTM output.
                Shape: [batch size, hidden dim * num directions]
            softmax_weights (torch.Tensor): The attention weights.
                Shape: [batch size, seq len]

        """
        attn_weights = self.attn(lstm_output).squeeze(2)
        softmax_weights = F.softmax(attn_weights, dim=1)
        softmax_weights = softmax_weights.unsqueeze(1)
        weighted_output = torch.bmm(softmax_weights, lstm_output).squeeze(1)
        return weighted_output, softmax_weights.squeeze(1)


class LSTM(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model for sequence classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the word embeddings.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        output_dim (int): The number of output classes.
        n_layers (int): The number of LSTM layers.
        bidirectional (bool): If True, the LSTM layers will be bidirectional.
        dropout_rate (float): The dropout rate to apply to the LSTM output.
        pad_index (int): The index used for padding sequences.

    Attributes:
        embedding (nn.Embedding): The embedding layer for word representations.
        lstm (nn.LSTM): The LSTM layer.
        fc (nn.Linear): The fully connected layer for classification.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x): Forward pass of the LSTM model.

    """

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
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): The input tensor of shape [batch size, sequence length].

        Returns:
            torch.Tensor: The output tensor of shape [batch size, output dim].

        """
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
    """
    LSTM model with attention mechanism.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the word embeddings.
        hidden_dim (int): The dimension of the hidden state of the LSTM.
        output_dim (int): The dimension of the output.
        n_layers (int): The number of LSTM layers.
        bidirectional (bool): Whether to use bidirectional LSTM.
        dropout_rate (float): The dropout rate.
        pad_index (int): The index used for padding sequences.

    Attributes:
        embedding (nn.Embedding): The embedding layer.
        lstm (nn.LSTM): The LSTM layer.
        attention (Attention): The attention mechanism.
        fc (nn.Linear): The fully connected layer.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x): Forward pass of the model.

    """

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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            prediction (torch.Tensor): Output tensor of shape (batch_size, output_dim).
            attn_weights (torch.Tensor): Attention weights tensor of shape (batch_size, sequence_length).

        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output = [batch size, seq len, hidden dim * num directions]

        attn_output, attn_weights = self.attention(lstm_output)
        # attn_output = [batch size, hidden dim * num directions]

        prediction = self.fc(self.dropout(attn_output))
        # prediction = [batch size, output dim]

        return prediction, attn_weights


class BERTClassifier(nn.Module):
    """
    BERTClassifier is a PyTorch module that uses the BERT transformer model for classification tasks.

    Args:
        transformer (nn.Module): The BERT transformer model.
        output_dim (int): The number of output classes.
        freeze (bool): Whether to freeze the parameters of the transformer model.

    Attributes:
        transformer (nn.Module): The BERT transformer model.
        fc (nn.Linear): The fully connected layer for classification.
    """

    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, tweet_token_ids, attention_mask=None):
        """
        Forward pass of the BERTClassifier.

        Args:
            tweet_token_ids (torch.Tensor): The input token IDs of the tweets.
            attention_mask (torch.Tensor, optional): The attention mask for the input tokens.

        Returns:
            torch.Tensor: The predicted class probabilities.
        """
        output = self.transformer(
            tweet_token_ids, attention_mask=attention_mask, output_attentions=True
        )
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction
