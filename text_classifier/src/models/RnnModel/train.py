import argparse
import os,sys
import numpy as np
import pandas as pd
import pickle

from src.models.RnnModel.model import RnnModel, get_keras_data
from src.models.RnnModel.config import *
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help='Path to training data csv')
    parser.add_argument('--model_dir', required=True, help='Path to models directory')
    args = parser.parse_args()
    train_data = args.train_data
    model_dir = args.model_dir

    df = pd.read_csv(train_data, encoding='latin-1')
    df = df.sample(frac=1).reset_index(drop=True)

    df[LABEL_COL] = df[LABEL_COL].map(label_dict)

    df_train, df_test = train_test_split(df, train_size=0.85)

    tok_raw = Tokenizer(oov_token=1)
    raw_text = np.hstack([df_train[DATA_COL].str.lower(), df_test[DATA_COL].str.lower()])
    tok_raw.fit_on_texts(raw_text)

    df_train[DATA_COL] = tok_raw.texts_to_sequences(df_train[DATA_COL].str.lower())
    df_test[DATA_COL] = tok_raw.texts_to_sequences(df_test[DATA_COL].str.lower())

    max_text_seq = np.max([np.max(df_train[DATA_COL].apply(lambda x: len(x))),
                           np.max(df_test[DATA_COL].apply(lambda x: len(x))),
                           ])

    print(f"max text seq:{max_text_seq}")

    max_text = np.max([np.max(df_train[DATA_COL].apply(max)),
                       np.max(df_test[DATA_COL].apply(max))]) + 2

    x_train = get_keras_data(df_train, max_text_seq)
    x_test = get_keras_data(df_test, max_text_seq)

    y_train = pd.get_dummies(df_train.label).values
    y_test = pd.get_dummies(df_test.label).values

    model = RnnModel().get_model([x_train[DATA_COL].shape[1]], max_text)

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    error, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Error: {error}, Test Accuracy: {accuracy}")

    with open(os.path.join(model_dir, TOKENIZER_NAME), 'wb') as handle:
        pickle.dump(tok_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(os.path.join(model_dir, RNN_MODEL_NAME))


if __name__ == "__main__":
    train()
