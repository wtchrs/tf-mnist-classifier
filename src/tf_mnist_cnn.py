from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import sys
import os


def plot_image(i, pred_ys, y_test, img):
    pred_ys, y_test, img = pred_ys[i], y_test[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(pred_ys)

    if predicted_label == y_test:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(predicted_label,
                                         100 * np.max(pred_ys),
                                         y_test), color=color)


def plot_value_array(i, pred_ys, true_label):
    pred_ys, true_label = pred_ys[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), pred_ys, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(pred_ys)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test_scaled = x_test.reshape(-1, 28, 28, 1) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    print(x_train[0].shape)

    if len(sys.argv) < 2 or sys.argv[1] != '1':
        model = load_model(os.path.dirname(__file__) + '\\best-cnn-model.h5')
    else:
        model = Sequential([Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)),
                            MaxPooling2D(2),
                            Conv2D(16, kernel_size=3, activation='relu', padding='same'),
                            MaxPooling2D(2),
                            Flatten(),
                            Dense(50, activation='relu'),
                            Dropout(0.3),
                            Dense(10, activation='softmax')
                            ])

        model.summary()

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        checkpoint_cb = ModelCheckpoint(os.path.dirname(__file__) + '\\best-cnn-model.h5',
                                        verbose=1, save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

        history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val),
                            callbacks=[checkpoint_cb, early_stopping_cb])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'])

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'])

        plt.show()

    print(model.evaluate(x_test_scaled, y_test))

    y_pred = model.predict(x_test_scaled)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols

    random_num = np.random.randint(len(x_test_scaled), size=num_images)
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for idx, num in enumerate(random_num):
        plt.subplot(num_rows, 2 * num_cols, 2 * idx + 1)
        plot_image(num, y_pred, y_test, x_test_scaled)
        plt.subplot(num_rows, 2 * num_cols, 2 * idx + 2)
        plot_value_array(num, y_pred, y_test)

    plt.show()
