import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(23)

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


# Create Model Object
array_size = 12
test_percent = 0.3
data_size = 1000
model = build_model()

# Create Synthetic Dataset to test model
#
# Goal: Create <data_size> arrays of the form [0,0,0,1,0,0,0,0,0], where the 1 is randomly selected
#
x_data = []
y_data = []
for i in range(data_size):
    temp = np.zeros(array_size)
    random_index = np.random.randint(array_size)
    temp[random_index] = 1  # randomly turn a 0 to 1
    temp = temp.reshape((-1,1))
    x_data.append(temp)
    # Arbitratily, if the first index is 1, 70% "win"
    win_p = np.random.uniform()
    if random_index == 1:
        if win_p > 0.3:
            y_data.append(1)
        else:
            y_data.append(0)
    elif (random_index == 0) or (random_index == 2):
        if win_p > 0.7:
            y_data.append(1)
        else:
            y_data.append(0)
    else:
        y_data.append(0)

x_data = np.array(x_data)
y_data = np.array(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_percent, random_state=1)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


num_epochs = 10
model.fit(X_train, y_train, epochs=num_epochs)
model.evaluate(X_test, y_test)

