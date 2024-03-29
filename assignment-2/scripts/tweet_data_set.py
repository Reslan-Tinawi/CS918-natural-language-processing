import torch
from torch.utils.data import Dataset


class TweetsDataset(Dataset):
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

        tweet_tensor = torch.tensor(self.vocab.lookup_indices(tweet), dtype=torch.long)
        tweet_label = self.label_encoder.transform([label]).squeeze()

        return tweet_id, tweet_tensor, tweet_label


class BERTTweetsDataset(Dataset):
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

        # tweet_tokens_ids = self.tokenizer(tweet, truncation=True)["input_ids"]

        tweet_label = self.label_encoder.transform([label]).squeeze()

        return tweet_id, tweet_tensor, tweet_mask, tweet_label
