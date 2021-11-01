import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.asarray(data["mfcc"], dtype=np.float32)
    y = np.asarray(data["labels"])
    return x, y


def prepare_datasets(test_size, validation_size):
    x, y = load_data(DATA_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # make 3d array
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # x_train = np.asarray(x_train, dtype=list).astype(np.float32)
    # x_validation = np.asarray(x_validation, dtype=object).astype(np.float32)
    # x_test = np.asarray(x_test, dtype=object).astype(np.float32)
    # y_train = np.asarray(y_train).astype(np.float32)
    # y_validation = np.asarray(y_validation).astype(np.float32)
    # y_test = np.asarray(y_test).astype(np.float32)

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))  # avoid overviting

    # Output layer
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model


if __name__ == "__main__":
    # create train, validation, and test set
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create CNN Model
    input_shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model

    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=30)

    # Evaluate CNN on test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accurasy on test set: {}".format(test_accuracy))