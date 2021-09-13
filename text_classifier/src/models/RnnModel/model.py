from keras.layers import Input, Dropout, Dense, BatchNormalization, GRU, Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from src.models.RnnModel.config import *


def get_keras_data(dataset, max_text_seq):
    keras_data = {
        DATA_COL: pad_sequences(dataset[DATA_COL], maxlen=max_text_seq)
    }
    return keras_data


class RnnModel:

    def get_model(self, shape, max_text):
        # Dropout rate
        dr_r = 0.15

        # Input
        text = Input(shape=shape, name="text")

        # Embeddings layer
        emb_text = Embedding(max_text, 50)(text)

        # RNN layer
        rnn_layer1 = GRU(64)(emb_text)

        main_l = rnn_layer1
        main_l = Dropout(dr_r)(Dense(64)(main_l))
        main_l = BatchNormalization()(main_l)
        main_l = Dropout(dr_r)(Dense(32)(main_l))
        main_l = BatchNormalization()(main_l)
        main_l = Dropout(dr_r)(Dense(16)(main_l))
        main_l = BatchNormalization()(main_l)

        # output
        output = Dense(NUM_CLASSES, activation="softmax")(main_l)

        # model
        model = Model([text], output)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
