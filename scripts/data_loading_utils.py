import pandas as pd
import numpy as np


def read_tweet_data(tweets_file_path: str) -> pd.DataFrame:
    """
    Reads tweet data from a file and returns it as a pandas DataFrame.

    Parameters:
        tweets_file_path (str): The file path of the tweet data file.

    Returns:
        pd.DataFrame: A DataFrame containing the tweet data.

    """
    data_list = []

    with open(tweets_file_path, encoding="utf8") as f:
        for line in f:
            fields = line.strip().split("\t")
            data_list.append(fields)

    df = pd.DataFrame(
        data=data_list,
        columns=[
            "tweet_id",
            "tweet_sentiment",
            "tweet_text",
        ],
    )

    return df


def load_embedding(embedding_file_path) -> dict[str, np.ndarray]:
    """
    Loads word embeddings from a file and returns a dictionary mapping words to their embedding vectors.

    Parameters:
        embedding_file (str): The path to the file containing word embeddings.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.

    """
    glove_embedding_dict = {}

    with open(embedding_file_path, encoding="utf8") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            word = tokens[0]
            word_embedding_vector = np.array(tokens[1:], dtype=np.float64)
            glove_embedding_dict[word] = word_embedding_vector

    return glove_embedding_dict
