import re

import contractions
import emoji
import nltk
import wordsegment
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict

# TODO: do we have to load every time?
wordsegment.load()


def normalize_with_without(tweet: str):
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"w/o", "without", tweet)
    return tweet


def remove_user_mentions(tweet: str):
    user_handle_pattern = re.compile(r"(@[a-zA-Z0-9_]+)")

    return user_handle_pattern.sub("", tweet)


def remove_tweet_hashtag(tweet: str):
    hashtag_pattern = re.compile(r"#(\w+)")

    return hashtag_pattern.sub("", tweet)


def remove_url(tweet: str):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    tweet = url_pattern.sub("", tweet)
    return tweet


def remove_special_characters(tweet: str):
    special_characters_pattern = re.compile(r"[^a-zA-Z0-9\s]")

    return special_characters_pattern.sub("", tweet)


def remove_digits(tweet: str):
    # remove digit with ordinal suffix
    digits_pattern = re.compile(r"\d+(st|nd|rd|th)?")

    return digits_pattern.sub("", tweet)


def remove_single_characters(tweet: str):
    single_characters_pattern = re.compile(r"\b\w\b")

    return single_characters_pattern.sub("", tweet)


def preprocess_tweet(tweet: str, tokenizer) -> list[str]:
    tweet = remove_url(tweet)  # what about emails?
    tweet = remove_user_mentions(tweet)
    tweet = remove_tweet_hashtag(tweet)
    tweet = remove_special_characters(tweet)
    tweet = remove_digits(tweet)
    tweet = remove_single_characters(tweet)
    # remove leading and trailing spaces
    tweet = tweet.strip()
    # remove multiple spaces
    tweet = re.sub(r"\s+", " ", tweet)

    # replace RT or rt with empty string
    tweet = re.sub(r"\bRT\b", "", tweet)

    # lowercase
    tweet = tweet.lower()

    tweet_tokens = tokenizer(tweet)

    return tweet_tokens


def clean_word(word: str) -> str:
    word = remove_special_characters(word)
    word = remove_single_characters(word)
    return word


def advanced_preprocessing(tweet: str, tokenizer):
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
        token for token in clean_text_tokens_list if len(token) > 1 and token not in stopwords
    ]

    # final_clean_text_tokens_list = []
    #
    # for idx, token in enumerate(clean_text_tokens_list):
    #     token = token.lower()
    #
    #     if not emoji_mask[idx]:
    #         # emoji don't have special characters, digits, or single characters
    #         # emoji is not in stopwords
    #         token = remove_special_characters(token)
    #         token = remove_digits(token)
    #         token = remove_single_characters(token)
    #         if token in stopwords.words("english"):
    #             continue
    #
    #     if len(token) > 1:
    #         final_clean_text_tokens_list.append(token)

    return " ".join(clean_text_tokens_list)
