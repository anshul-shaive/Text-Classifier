import numpy as np
import pandas as pd
import pickle
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.models.RFModel.config import *


def get_use_features(df, get_use_embed):
    df_use_embed = pd.DataFrame(columns=range(512))
    for i in range(df.shape[0]):
        use_embeddings = get_use_embed(np.array([df.text[i]]))
        df_use_embed = df_use_embed.append(pd.Series(use_embeddings.numpy().ravel()), ignore_index=True)
    return df_use_embed


class RFModel:
    def __init__(self):
        self.use_model_version = '4'
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/" + self.use_model_version
        self.get_use_embed = hub.load(self.module_url)

    def train(self):
        df = pd.read_csv("data\\train_set.csv", encoding='latin-1')
        df = df.sample(frac=1).reset_index(drop=True)
        df[LABEL_COL] = df[LABEL_COL].map(label_dict)
        df_use_embed = get_use_features(df, self.get_use_embed)
        df = pd.concat([df, df_use_embed], axis=1)
        df_train, df_test = train_test_split(df, train_size=0.85)
        model = RandomForestClassifier()
        model.fit(df_train.iloc[:, 2:], df_train[LABEL_COL])
        score = model.score(df_test.iloc[:, 2:], df_test[LABEL_COL])
        print(f'Random Forests Model Accuracy: {score}')

        with open('model\\rf_model.pkl', 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    RFModel().train()
