import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def plot_wordcloud(wordcloud, title: str):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def generate_ngram_frequencies(
    corpus: list[str], n_grams: int, max_features
) -> dict[str, int]:
    vectorizer = CountVectorizer(
        ngram_range=(n_grams, n_grams), stop_words="english", max_features=max_features
    )
    X = vectorizer.fit_transform(corpus)
    ngram_array = X.toarray()
    sum_ngrams = ngram_array.sum(axis=0)
    ngram_list = vectorizer.get_feature_names_out()
    ngram_freq = dict(zip(ngram_list, sum_ngrams))
    return ngram_freq


def generate_wordcloud_with_ngrams(
    ngram_freq: dict[str, int], n_grams: int, wordcloud_title: str
):

    # Load the twitter logo image
    twitter_mask = np.array(Image.open("twitter_mask.png"))

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=800,
        max_words=100,
        background_color="white",
        mask=twitter_mask,
        contour_width=3,
        contour_color="steelblue",
    ).generate_from_frequencies(ngram_freq)

    # Display the generated image
    plot_wordcloud(wordcloud, f"{wordcloud_title} - {n_grams}-grams")


def plot_top_common_ngrams(n_gram_freq_dict_list: list[dict[str, int]]):
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    for idx, n_gram_freq_dict in enumerate(n_gram_freq_dict_list):
        n_gram_freq_df = pd.DataFrame(
            n_gram_freq_dict.items(), columns=["n_gram", "frequency"]
        ).sort_values(by="frequency", ascending=False)
        n_gram_freq_df = n_gram_freq_df.head(10)

        # color by frequency
        cmap = plt.cm.Reds
        norm = plt.Normalize(
            n_gram_freq_df["frequency"].min(), n_gram_freq_df["frequency"].max()
        )
        colors = cmap(norm(n_gram_freq_df["frequency"]))

        # create bar plot and color by frequency
        ax[idx].barh(
            n_gram_freq_df["n_gram"], n_gram_freq_df["frequency"], color=colors
        )
        ax[idx].invert_yaxis()  # labels read top-to-bottom
        ax[idx].set_xlabel("Frequency")
        ax[idx].set_title(f"Top 10 {idx + 1}-grams")
        ax[idx].set_ylabel(f"{idx + 1}-gram")

    plt.tight_layout()
    plt.show()
