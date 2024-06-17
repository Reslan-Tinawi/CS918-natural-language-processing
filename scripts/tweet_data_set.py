import torch
from torch.utils.data import Dataset


class TweetsDataset(Dataset):
    """
    A custom dataset class for handling tweet data.

    Args:
        tweet_ids (list): List of tweet IDs.
        tweets (list): List of tweet texts.
        labels (list): List of tweet labels.
        vocab (Vocab): Vocabulary object for tokenization.
        label_encoder (LabelEncoder): LabelEncoder object for label encoding.

    Returns:
        tuple: A tuple containing the tweet ID, tokenized tweet tensor, and label.

    """

    def __init__(self, tweet_ids, tweets, labels, vocab, label_encoder):
        self.tweet_ids = tweet_ids
        self.tweets = tweets
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet_id = self.tweet_ids[idx]
        tweet = self.tweets[idx]
        label = self.labels[idx]

        tweet_tokens = tweet.split()

        tweet_tensor = torch.tensor(
            self.vocab.lookup_indices(tweet_tokens), dtype=torch.long
        )

        tweet_label = self.label_encoder.transform([label]).squeeze()

        return tweet_id, tweet_tensor, tweet_label


class BERTTweetsDataset(Dataset):
    """
    A PyTorch dataset class for processing tweets using BERT.

    Args:
        tweet_ids (list): List of tweet IDs.
        tweets (list): List of tweets.
        labels (list): List of labels.
        tokenizer: The BERT tokenizer.
        label_encoder: The label encoder.

    Returns:
        tuple: A tuple containing the tweet ID, tweet tensor, tweet mask, and tweet label.
    """

    def __init__(self, tweet_ids, tweets, labels, tokenizer, label_encoder):
        self.tweet_ids = tweet_ids
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet_id = self.tweet_ids[idx]
        tweet = self.tweets[idx]
        label = self.labels[idx]

        tweet_dict = self.tokenizer.encode_plus(
            tweet, padding="max_length", max_length=128, truncation=True
        )
        tweet_tensor = torch.tensor(tweet_dict["input_ids"], dtype=torch.long)
        tweet_mask = torch.tensor(tweet_dict["attention_mask"], dtype=torch.long)

        tweet_label = self.label_encoder.transform([label]).squeeze()

        return tweet_id, tweet_tensor, tweet_mask, tweet_label
