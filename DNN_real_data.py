import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main():
    n0523_df = pd.read_csv('./data/n0523.csv')
    data = n0523_df.values

    sample = (data[63000:100000])
    sample[np.where(sample == 1)[0]] = 0
    sample[np.where(sample == 2)[0]] = 0

    model = build_model()
    
    # Build X data slices from flat array:
    n_slice = 12
    x_data = []
    for i in range(len(sample)-n_slice+1):
        x_data.append(sample[i:(i+n_slice)])

    # Create y_data for expected output    
    y_data = create_expected(sample[:len(sample)-n_slice+1])

    # Ensure everything is in numpy
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split into testing and training data
    test_percent = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_percent, random_state=1)

    # Train & Evaluate Model:
    num_epochs = 10
    model.fit(X_train, y_train, epochs=num_epochs)
    print(model.evaluate(X_test, y_test))
    #0.983238697052002 accuracy


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(12,1)))
    model.add(tf.keras.layers.Dense(24, activation = "relu"))
    model.add(tf.keras.layers.Dense(12, activation = "relu"))
    model.add(tf.keras.layers.Dense(2, activation = "relu"))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['sparse_categorical_accuracy'])
    return model


def create_expected(sample):
    expected = np.zeros(len(sample))
    space = 0
    cooldown = 0
    values = []
    indexes = []
    for i in range(len(sample)):
        if sample[i] == 0 or cooldown > 0:
            space += 1
        else:
            values.append(sample[i])
            indexes.append(i)

        if cooldown > 0:
            cooldown -= 1

        if space < 15:
            continue
        elif len(values) > 0 and sample[i] == 0:
            expected[ indexes[np.where(values == np.max(values))[0][0]] ] = 1
            values = []
            indexes = []
            space = 0
            cooldown = 20
    return expected

if __name__ == '__main__':
    main()