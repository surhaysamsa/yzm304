import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, x):
        # x: (input_size,) -> (input_size, 1)
        x = x.reshape(self.input_size, 1)
        h = np.tanh(np.dot(self.Wxh, x) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y, h

    def predict(self, inputs):
        y, _ = self.forward(inputs)
        return int(y > 0)

# Training ve loss function  train_and_evaluate.py nin içinde
