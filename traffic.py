import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def lr_schedule(epoch):
    """
    Sets the learning rate as a decreasing function of the epoch 
    """
    return 0.001 * (0.1 ** int(epoch / 10))


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Get model summary
    model.summary()

    # Fit model on training data
    model.fit(
        x_train, y_train, 
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[
            LearningRateScheduler(lr_schedule),
            ModelCheckpoint('model.h5', save_best_only=True)
            ]
        )

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = list()
    for category in range(NUM_CATEGORIES):
        DIR = os.path.join(data_dir, str(category))
        for name in os.listdir(DIR):

            # Retrieve image
            path = os.path.join(DIR, name)
            img = cv2.imread(path)

            # Resize the image
            dsize = (IMG_WIDTH, IMG_HEIGHT)
            img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

            # Normalize data so that is included between 0 and 1
            img = img / 255.0

            # Finish dataset
            images.append(img)
            labels.append(category)
            
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Define input_shape
    CHANNELS = 3
    input_shape = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)
    
    # Create a sequential neural network
    model = tf.keras.models.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=input_shape),
        Conv2D(16, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, activation='relu'),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 2, activation='relu'),
        Conv2D(64, 2, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
