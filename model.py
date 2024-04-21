import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model
