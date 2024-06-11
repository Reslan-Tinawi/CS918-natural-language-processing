import re

import contractions
import emoji
import nltk
import wordsegment
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict

wordsegment.load()


def normalize_with_without(tweet: str):
    """
    Normalizes the text by replacing 'w/' with 'with' and 'w/o' with 'without'.

    Args:
        tweet (str): The input tweet to be normalized.

    Returns:
        str: The normalized tweet.
    """
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"w/o", "without", tweet)
    return tweet


def remove_user_mentions(tweet: str):
    """
    Removes user mentions from a given tweet.

    Args:
        tweet (str): The tweet text.

    Returns:
        str: The tweet text with user mentions removed.
    """
    user_handle_pattern = re.compile(r"(@[a-zA-Z0-9_]+)")

    return user_handle_pattern.sub("", tweet)


def remove_tweet_hashtag(tweet: str):
    """
    Removes hashtags from a given tweet.

    Args:
        tweet (str): The tweet to remove hashtags from.

    Returns:
        str: The tweet with hashtags removed.
    """
    hashtag_pattern = re.compile(r"#(\w+)")

    return hashtag_pattern.sub("", tweet)


def remove_url(tweet: str):
    """
    Removes URLs from a given tweet.

    Args:
        tweet (str): The input tweet containing URLs.

    Returns:
        str: The tweet with URLs removed.
    """
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    tweet = url_pattern.sub("", tweet)
    return tweet


def remove_special_characters(tweet: str):
    """
    Removes special characters from a given tweet.

    Args:
        tweet (str): The tweet to remove special characters from.

    Returns:
        str: The tweet with special characters removed.
    """
    special_characters_pattern = re.compile(r"[^a-zA-Z0-9\s]")

    return special_characters_pattern.sub("", tweet)


def remove_digits(tweet: str):
    """
    Removes digits from a given tweet.

    Parameters:
        tweet (str): The tweet to remove digits from.

    Returns:
        str: The tweet with digits removed.
    """
    digits_pattern = re.compile(r"\d+(st|nd|rd|th)?")

    return digits_pattern.sub("", tweet)


def remove_single_characters(tweet: str):
    """
    Removes single characters from a given tweet.

    Args:
        tweet (str): The input tweet.

    Returns:
        str: The tweet with single characters removed.
    """
    single_characters_pattern = re.compile(r"\b\w\b")

    return single_characters_pattern.sub("", tweet)


def clean_word(word: str) -> str:
    """
    Cleans a word by removing special characters and single characters.

    Args:
        word (str): The word to be cleaned.

    Returns:
        str: The cleaned word.
    """
    word = remove_special_characters(word)
    word = remove_single_characters(word)
    return word


def preprocess_tweet(tweet: str, tokenizer):
    """
    Preprocesses a tweet by performing various text cleaning operations.

    Args:
        tweet (str): The input tweet to be preprocessed.
        tokenizer: The tokenizer object used to tokenize the tweet.

    Returns:
        str: The preprocessed tweet as a string.

    """
    # start by removing user mentions and urls
    tweet = tweet.lower()
    tweet = normalize_with_without(tweet)
    tweet = remove_user_mentions(tweet)
    tweet = remove_url(tweet)
    tweet = remove_digits(tweet)
    tweet = tweet.strip()
    tweet = re.sub(r"\s+", " ", tweet)
    tweet = re.sub(r"\bRT\b", "", tweet)
    tweet = re.sub("ac/dc", "acdc", tweet)

    # fix contractions
    tweet = contractions.fix(tweet)

    # tokenize the tweet
    tokens_list = tokenizer.tokenize(tweet)

    clean_text_tokens_list = []

    for token in tokens_list:
        if token in emoticons:
            emoticon_text = emoticons[token]
            clean_text_tokens_list.append(emoticon_text)
        elif emoji.is_emoji(token):
            emoji_text = emoji.demojize(token, delimiters=("<", ">"))
            clean_text_tokens_list.append(emoji_text)
        elif token.startswith("#"):
            token_segment_list = wordsegment.segment(token)
            token_segment_list_cleaned = [
                clean_word(word) for word in token_segment_list
            ]
            clean_text_tokens_list.extend(token_segment_list_cleaned)
        elif token in slangdict:
            cleaned_token = slangdict[token]
            clean_text_tokens_list.append(cleaned_token)
        else:
            cleaned_token = clean_word(token)
            clean_text_tokens_list.append(cleaned_token)

    # remove empty strings and stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))
    clean_text_tokens_list = [
        token.lower()
        for token in clean_text_tokens_list
        if len(token) > 1 and token not in stopwords
    ]

    return " ".join(clean_text_tokens_list)
