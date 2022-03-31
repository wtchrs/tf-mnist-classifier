from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

    plt.xlabel('{} {:2.0f}% ({})'.format(labels[predicted_label],
                                         100 * np.max(pred_ys),
                                         labels[y_test[0]]), color=color)


def plot_value_array(i, pred_ys, true_label):
    pred_ys, true_label = pred_ys[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), pred_ys, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(pred_ys)
    true_label = true_label[0]

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # fig = plt.figure(figsize=(20, 5))
    # for i in range(36):
    #     ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    #     ax.imshow(np.squeeze(x_train[i]))
    # fig.show()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test)

    print('x_train shape: ', x_train.shape)

    print('number of training data: ', x_train.shape[0])
    print('number of test data: ', x_test.shape[0])
    print('number of validation data', x_val.shape[0])

    if len(sys.argv) < 2 or sys.argv[1] != '1':
        model = load_model(os.path.dirname(__file__) + '\\best-cifar10-cnn.h5')
    else:
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax', name='Output'))

        model.summary()

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        checkpoint_cb = ModelCheckpoint(os.path.dirname(__file__) + '\\best-cifar10-cnn.h5',
                                        verbose=1, save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

        history = model.fit(x_train, y_train, batch_size=64, epochs=100,
                            validation_data=(x_val, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

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

    print(model.evaluate(x_test, y_test))

    y_pred = model.predict(x_test)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols

    random_num = np.random.randint(len(x_test), size=num_images)
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for idx, num in enumerate(random_num):
        plt.subplot(num_rows, 2 * num_cols, 2 * idx + 1)
        plot_image(num, y_pred, y_test, x_test)
        plt.subplot(num_rows, 2 * num_cols, 2 * idx + 2)
        plot_value_array(num, y_pred, y_test)

    plt.show()
