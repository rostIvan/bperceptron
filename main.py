#! /usr/bin/env python3
import math
from dataclasses import dataclass
from typing import List, Callable

from tabulate import tabulate

from misc import dot_vectors, get_truth_table


@dataclass
class PredictionModel:
    weights: list
    inputs: tuple
    activation_function: Callable[[int], int]

    def predict(self) -> int:
        return self.activation_function(dot_vectors(self.inputs, self.weights))


class BinaryPerceptron:
    default_weights = [0.1, 0.1, 0.1]
    threshold = 1.7

    def __init__(self, input_layer: List[tuple]) -> None:
        self.input_layer = input_layer
        self.weights: List[float] = self.default_weights
        self.on_weights_change = None
        self.counter = 0
        self.weights_shift_counter = 0

    def activation_function(self, x) -> int:
        return 1 if x >= self.threshold else 0

    def train(self, learning_rate=0.7):
        for x1, x2, x3, expected in self.input_layer:
            m = PredictionModel(weights=self.weights,
                                inputs=(x1, x2, x3),
                                activation_function=self.activation_function)
            prediction = m.predict()
            if prediction != expected:
                self.correct_weights(inputs=(x1, x2, x3),
                                     actual=prediction,
                                     expected=expected,
                                     learning_rate=learning_rate)
                self.weights_shift_counter += 1
            self.counter += 1

    def predict(self, x1, x2, x3) -> int:
        m = PredictionModel(weights=self.weights,
                            inputs=(x1, x2, x3),
                            activation_function=self.activation_function)
        return m.predict()

    def correct_weights(self, inputs, actual, expected, learning_rate):
        def d_i(x_i): return (expected - actual) * x_i * learning_rate

        before = self.weights
        self.weights = [w_i + d_i(x_i)
                        for w_i, x_i in zip(self.weights, inputs)]

        self.on_weights_change(before, self.weights)
        self.train(learning_rate)

    def weights_change_listener(self, callback: Callable[[list, list], None]):
        self.on_weights_change = callback


if __name__ == '__main__':
    truth_table = get_truth_table()
    model = BinaryPerceptron(input_layer=truth_table)
    model.weights_change_listener(lambda w1, w2: print(w1, '->', w2))
    model.train(learning_rate=0.06)

    fields = ['inputs', 'weights', 'actual', 'expected']
    output = [((x1, x2, x3),
               [round(w, 4) for w in model.weights],
               model.predict(x1, x2, x3),
               expected)
              for x1, x2, x3, expected in truth_table]
    print(f'\n{tabulate(output, headers=fields)}')
    print(f'\nTook {model.counter} iterations, '
          f' Weights shifts {model.weights_shift_counter}')
