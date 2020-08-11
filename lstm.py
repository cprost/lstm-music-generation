import parameters as params

import tensorflow as tf

class LSTM(tf.keras.model):
    def __init__(self, output_size):
        super().__init__()
        self.input_layer = tf.keras.layers.LSTM(
            units=512,
            return_sequences=True,
            input_shape=(
                params.SEQ_LEN,
                params.SEQ_WIDTH
            )
        )
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.lstm_inner = tf.keras.layers.LSTM(512)
        self.dense = tf.keras.layers.Dense(output_size)
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.dropout(x)
        x = self.lstm_inner(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.activation(x)
        
        return x