#! /usr/bin/env python3
from dataclasses import dataclass
from typing import List

from tabulate import tabulate

from misc import multiply_vectors, get_truth_table


@dataclass
class BinaryPerceptron:
    weights: list
    inputs: tuple

    threshold: float
    learning_rate: float
    T: int  # expected

    @property
    def a(self) -> float:
        return multiply_vectors(self.inputs, self.weights)

    # actual
    @property
    def Y(self) -> int:
        return 1 if self.a >= self.threshold else 0

    @property
    def delta(self) -> int:
        return self.T - self.Y

    def correct_weights(self):
        def d_i(x_i): return self.learning_rate * self.delta * x_i

        self.weights = [w_i + d_i(x_i)
                        for w_i, x_i in zip(self.weights, self.inputs)]

    def __str__(self) -> str:
        return f'BP(ws={self.weights}, ' \
               f'xs={self.inputs}, ' \
               f'a={self.a}, ' \
               f'Y={self.Y},' \
               f'T={self.T})'


class BinaryPerceptronModel:
    default_weights = [0.02, 0.01, 0.02]
    threshold = 0.5

    def __init__(self, input_layer: List[tuple]) -> None:
        self.input_layer = input_layer
        self.bps = None

    def train(self, learning_rate=0.7) -> List[BinaryPerceptron]:
        self.bps = self._initial_train(learning_rate)
        while self._bad_trained():
            self._train()
        return self.bps

    def _initial_train(self, learning_rate) -> List[BinaryPerceptron]:
        return [BinaryPerceptron(self.default_weights,
                                 (x1, x2, x3),
                                 self.threshold,
                                 learning_rate,
                                 expected)
                for x1, x2, x3, expected in self.input_layer]

    def _bad_trained(self):
        return any(df_i.delta != 0 for df_i in self.bps)

    def _train(self):
        for p in self.bps:
            if p.T != p.Y:
                p.correct_weights()


if __name__ == '__main__':
    model = BinaryPerceptronModel(input_layer=get_truth_table())
    train = model.train(learning_rate=0.06)

    table = [(m.weights,
              m.inputs,
              m.a,
              m.threshold,
              m.Y,
              m.T)
             for m in train]
    print(tabulate(table, headers=['weights',
                                   'inputs',
                                   'a',
                                   'threshold', 'Y', 'T']))
