import pickle
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from src.models.RnnModel.config import *
from src.models.RnnModel.model import get_keras_data


class RnnModelInference:

    def __init__(self):
        with open(f'model\\{TOKENIZER_NAME}', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.model = keras.models.load_model(f'model\\{RNN_MODEL_NAME}')

    def make_prediction(self, x_test):
        predictions = self.model.predict(x_test)
        labels = np.argmax(predictions, axis=-1)
        reverse_label_dict = {v: k for k, v in label_dict.items()}
        predicted_classes = pd.Series(labels).map(reverse_label_dict)
        return predicted_classes

    def api_inference(self, text):
        df = pd.DataFrame({DATA_COL: text})
        df[DATA_COL] = self.tokenizer.texts_to_sequences(df[DATA_COL].str.lower())
        padded_text = get_keras_data(df, MAX_TEXT_SEQ)
        predicted_class = self.make_prediction(padded_text)
        return list(predicted_class)

    def csv_inference(self):
        df_test = pd.read_csv('data\\test_set.csv')
        original_text = df_test.text.copy()
        df_test[DATA_COL] = self.tokenizer.texts_to_sequences(df_test[DATA_COL].str.lower())
        x_test = get_keras_data(df_test, MAX_TEXT_SEQ)
        predicted_classes = self.make_prediction(x_test)
        pd.DataFrame({DATA_COL: original_text, LABEL_COL: predicted_classes}).to_csv('data\\test_labels.csv',
                                                                                     index=False)


if __name__ == "__main__":
    RnnModelInference().csv_inference()
