import numpy as np
from config import BRAIN_INPUTS, BRAIN_HIDDEN, BRAIN_OUTPUTS, MUTATION_RATE, MUTATION_STRENGTH


class Brain:
    """
    A minimal feedforward neural network.
    Architecture: inputs -> hidden(tanh) -> outputs(tanh)
    Weights ARE the genome — no backprop, only mutation.
    """

    def __init__(self, weights=None):
        if weights is None:
            # Random initialization in [-1, 1]
            self.W1 = np.random.uniform(-1, 1, (BRAIN_HIDDEN, BRAIN_INPUTS))
            self.b1 = np.random.uniform(-0.5, 0.5, (BRAIN_HIDDEN,))
            self.W2 = np.random.uniform(-1, 1, (BRAIN_OUTPUTS, BRAIN_HIDDEN))
            self.b2 = np.random.uniform(-0.5, 0.5, (BRAIN_OUTPUTS,))
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def forward(self, sensor_inputs: np.ndarray) -> np.ndarray:
        """
        sensor_inputs: 1D array of length BRAIN_INPUTS
        returns: 1D array of length BRAIN_OUTPUTS in [-1, 1]
        """
        h = np.tanh(self.W1 @ sensor_inputs + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)
        return out

    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def mutate(self):
        """Return a new Brain with Gaussian noise applied to weights."""
        W1 = self.W1.copy()
        b1 = self.b1.copy()
        W2 = self.W2.copy()
        b2 = self.b2.copy()

        for arr in (W1, b1, W2, b2):
            mask = np.random.rand(*arr.shape) < MUTATION_RATE
            arr += mask * np.random.normal(0, MUTATION_STRENGTH, arr.shape)
            np.clip(arr, -3, 3, out=arr)  # prevent runaway weights

        return Brain((W1, b1, W2, b2))

    def clone(self):
        return Brain(self.get_weights())
