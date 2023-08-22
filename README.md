# Neural Network Library with NumPy

Welcome to the Neural Network Library with NumPy! This library enables you to create, train, and evaluate neural networks using the power of the NumPy library.

## Features

- Simple implementation of neural networks with fully connected layers.
- Support for training with backpropagation using optimization algorithms.
- Customizable network architecture by adding various layers.
- Common activation functions such as ReLU, sigmoid, and more.
- Implementation of layers including LSTM, dense (fully connected), convolutional, pooling, embedding, dropout, and flatten.
- Code examples to help you get started quickly.

## Usage

Here's how you can create and train a simple neural network using this library:

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Model() 
model.add(Dense(input_shape=(784,),output_shape=100))
model.add(Sigmoid())
model.add(Dense(100))
model.add(Sigmoid())
model.add(Dense(10))
model.add(Sigmoid())
model.compile(BinaryCrossEntropy())

model.fit(
    (X_train, y_train),
    (X_test, y_test),
    epoch=10,
    learning_rate=0.1,
    batch_size=64
    )
```