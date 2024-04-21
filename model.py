import keras
from keras import Sequential
from keras.src.layers import Dense

def create_model():
    model = Sequential()
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model