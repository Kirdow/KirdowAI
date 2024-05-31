import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers as op
import random as rng

def validate_devices():
    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)

    gpu_available = tf.config.list_physical_devices('GPU')
    print("GPU Available:", gpu_available)

    exit(0)

# validate_devices()

training_size = 100_000
num_max = 1_000

print("Creating model layers")

input_nodes = 16 # because 16-bit
output_nodes = 2
hidden_layers = [64, 16, 4]

model = keras.Sequential()
model.add(layers.Input(shape=(input_nodes,)))

for units in hidden_layers:
    model.add(layers.Dense(units, activation='relu'))

model.add(layers.Dense(output_nodes, activation='softmax'))

print("Compiling model")

model.compile(optimizer=op.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

def get_num(cap=None):
    rng_cap = num_max if cap is None else cap
    return rng.randint(1, rng_cap)

def get_num_bin(n):
    if not 1 <= n < 2 ** 15:
        raise ValueError("Input out of range for positive signed 16-bit integer")

    binary_str = f"{n & (2**16-1):016b}"
    return [int(bit) for bit in binary_str]

def get_expected(n):
    n %= 3
    if n == 1: return [1, 0]
    elif n == 2: return [0, 1]
    return 'undefined'

def parse_input_output_data(N=None):
    N = training_size if N is None else N

    inputs = []
    outputs = []

    percent = 0
    index = 0
    while index < N:
        num = get_num()
        mod3 = num % 3
        if mod3 == 0:
            continue
        inputs.append(get_num_bin(num))
        outputs.append([1, 0] if mod3 == 1 else [0, 1])
        index += 1
        new_percent = index * 100 // N
        if new_percent != percent:
            print(f" {new_percent}%\r", end='',flush=True)
        percent = new_percent

    return np.array(inputs), np.array(outputs)

print("Creating data")
input_data, output_data = parse_input_output_data()

print("Training with data")
model.fit(input_data, output_data, epochs=10, batch_size=None)

def filter_prediction(a):
    return [round(n, 3) for n in a]

print("Predicting...")
predict_index = 0
while predict_index < 15:
    num = get_num()
    if predict_index < 9:
        if num % 3 == 0:
            continue
    else:
        if num % 3 != 0:
            continue

    prediction = model.predict(np.array([get_num_bin(num)]))[0]
    print(f"{predict_index}: {num} % 3 plot = [{', '.join(map(str, filter_prediction(prediction)))}] | expected: {get_expected(num)}")
    predict_index += 1

print("Done!")