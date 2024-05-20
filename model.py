import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten
from tensorflow.keras import layers

"""
        conv2d
     /           \
input                 concat-  layer - otp 
     \               /
       flatten - layer 
    
"""


def create_model():
    input_shape = (256, 256, 3)

    model = Sequential()

    '''model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))'''

    #model.summary()

    model.add(Flatten())
    model.add(Dense(80))
    model.add(Dense(80))
    model.add(Dense(80))
    model.add(Dense(32))
    model.add(Dense(10))
    model.add(Dense(4, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=opt,
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model
