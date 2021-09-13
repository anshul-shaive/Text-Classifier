import pickle
import pandas as pd

from src.models.RFModel.config import *
from src.models.RFModel.model import RFModel, get_use_features


class RFModelInference:

    def __init__(self):
        with open('model\\rf_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.get_use_embed = RFModel().get_use_embed

    def make_prediction(self, x_test):
        predictions = self.model.predict(x_test)
        reverse_label_dict = {v: k for k, v in label_dict.items()}
        predicted_classes = pd.Series(predictions).map(reverse_label_dict)
        return predicted_classes

    def api_inference(self, text):
        df = pd.DataFrame({DATA_COL: text})
        df_use_embed = get_use_features(df, self.get_use_embed)
        df = pd.concat([df, df_use_embed], axis=1)
        predicted_classes = self.make_prediction(df.iloc[:, 1:])
        return list(predicted_classes)

    def csv_inference(self):
        df_test = pd.read_csv('data\\test_set.csv')
        df_use_embed = get_use_features(df_test, self.get_use_embed)
        df_test = pd.concat([df_test, df_use_embed], axis=1)
        predicted_classes = self.make_prediction(df_test.iloc[:, 1:])
        pd.DataFrame({DATA_COL: df_test.text, LABEL_COL: predicted_classes}).to_csv('data\\test_labels_rf.csv',
                                                                                    index=False)


if __name__ == "__main__":
    RFModelInference().csv_inference()
