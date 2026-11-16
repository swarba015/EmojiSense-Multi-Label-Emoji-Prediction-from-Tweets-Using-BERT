from google.colab import drive
drive.mount('/content/drive')

import emoji
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
import os
import re
import swifter
from multiprocessing import Pool, cpu_count
import numpy as np


def contains_emoji(text):
    return any(char in emoji.EMOJI_DATA for char in str(text))




folder_path = '/content/drive/My Drive/nlp/emoji dataset - Copy'
all_dfs = []

for i in range(1, 44):
    file_path = os.path.join(folder_path, f'{i}.csv')
    try:
        df = pd.read_csv(file_path)
        df['ds'] = i
        all_dfs.append(df)
    except pd.errors.ParserError as e:
        print(f"Error reading file {file_path}: {e}")


merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates()




# Ensure stopwords are downloaded
nltk.download('stopwords')

# Initialize tokenizer and preprocessing tools
tweet_tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def clean_emoji_tweet(tweet):
    """
    Cleans tweets by:
    - Tokenizing using TweetTokenizer
    - Lowercasing tokens
    - Removing stopwords, hashtags, and punctuation
    - Preserving emojis and text emojis
    """
    tokens = tweet_tokenizer.tokenize(tweet)

    filtered = [
        token.lower() for token in tokens
        if token.lower() not in stop_words
        and not token.startswith("#")
        and token not in punctuation
    ]

    return ' '.join(filtered)



def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

def apply_clean(df):
    df['tokenized_text'] = df['Text'].apply(clean_emoji_tweet)
    return df

merged_df = parallelize_dataframe(merged_df, apply_clean, n_cores=cpu_count())


from emoji import EMOJI_DATA

def extract_emojis(text):
    """
    Extract emojis from a string.
    """
    return [char for char in str(text) if char in EMOJI_DATA]




assert 'ds' in merged_df.columns

available_ds = merged_df['ds'].unique()

# Initialize empty lists for each split
train_parts = []
dev_parts = []
test_parts = []

# Proportional split for each dataset
for d in available_ds:
    subset = merged_df[merged_df['ds'] == d]
    subset = subset.sample(frac=1, random_state=42)  # shuffle

    n = len(subset)
    n_train = int(n * 0.70)
    n_dev = int(n * 0.15)

    train_parts.append(subset[:n_train])
    dev_parts.append(subset[n_train:n_train + n_dev])
    test_parts.append(subset[n_train + n_dev:])

# Concatenate parts to final DataFrames
train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
dev_df = pd.concat(dev_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)



# O Save them
train_df.to_pickle("train_balanced.pkl")
dev_df.to_pickle("dev_balanced.pkl")
test_df.to_pickle("test_balanced.pkl")
