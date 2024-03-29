import pandas as pd


def read_tweet_data(tweets_file_path: str):
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


def load_embedding(embedding_file):
    glove_embedding_dict = {}

    with open(embedding_file, encoding="utf8") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            word = tokens[0]
            word_embedding_vector = np.array(tokens[1:], dtype=np.float64)
            glove_embedding_dict[word] = word_embedding_vector

    return glove_embedding_dict
