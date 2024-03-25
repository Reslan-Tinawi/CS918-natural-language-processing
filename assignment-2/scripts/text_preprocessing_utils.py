import re


def remove_user_mentions(tweet: str):
    user_handle_pattern = re.compile("(@[a-zA-Z0-9_]+)")

    return user_handle_pattern.sub("", tweet)


def remove_tweet_hashtag(tweet: str):
    hashtag_pattern = re.compile("#(\w+)")

    return hashtag_pattern.sub("", tweet)


def remove_url(tweet: str):
    url_pattern = re.compile(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    tweet = url_pattern.sub("", tweet)
    return tweet


def remove_special_characters(tweet: str):
    special_characters_pattern = re.compile("[^a-zA-Z0-9\s]")

    return special_characters_pattern.sub("", tweet)


def remove_digits(tweet: str):
    digits_pattern = re.compile(r"\b\d+\b")

    return digits_pattern.sub("", tweet)


def remove_single_characters(tweet: str):
    single_characters_pattern = re.compile(r"\b\w{1}\b")

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

    # lowercase
    tweet = tweet.lower()

    tweet_tokens = tokenizer(tweet)

    return tweet_tokens
