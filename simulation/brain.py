import numpy as np
from config import BRAIN_INPUTS, BRAIN_HIDDEN_LAYERS, BRAIN_OUTPUTS, MUTATION_RATE, MUTATION_STRENGTH


class Brain:
    """
    Feedforward neural network with a configurable number of hidden layers.

    Architecture is driven by BRAIN_HIDDEN_LAYERS in config:
        []         → inputs directly to outputs (linear, no hidden layer)
        [8]        → one hidden layer of 8 neurons  (original behaviour)
        [8, 16, 6] → three hidden layers: 8 → 16 → 6

    All hidden activations: tanh.  Output activation: tanh.
    Weights ARE the genome — no backprop, mutation only.
    """

    def __init__(self, weights=None):
        # Build layer size list: [inputs, h1, h2, ..., outputs]
        self._sizes = [BRAIN_INPUTS] + list(BRAIN_HIDDEN_LAYERS) + [BRAIN_OUTPUTS]

        if weights is None:
            self._Ws = []
            self._bs = []
            for i in range(len(self._sizes) - 1):
                fan_in  = self._sizes[i]
                fan_out = self._sizes[i + 1]
                self._Ws.append(np.random.uniform(-1, 1, (fan_out, fan_in)))
                self._bs.append(np.random.uniform(-0.5, 0.5, (fan_out,)))
        else:
            # weights is a tuple of (Ws_list, bs_list)
            self._Ws, self._bs = weights
            # Ensure they are proper numpy arrays (important after npz reload)
            self._Ws = [np.array(w) for w in self._Ws]
            self._bs = [np.array(b) for b in self._bs]

    def forward(self, sensor_inputs: np.ndarray) -> np.ndarray:
        """
        sensor_inputs: 1D float32 array of length BRAIN_INPUTS
        returns:       1D array of length BRAIN_OUTPUTS in [-1, 1]
        """
        x = sensor_inputs
        for i, (W, b) in enumerate(zip(self._Ws, self._bs)):
            x = np.tanh(W @ x + b)
        return x

    def get_weights(self):
        """Return a deep-copy tuple of (Ws, bs) lists — safe to store."""
        return ([W.copy() for W in self._Ws],
                [b.copy() for b in self._bs])

    def mutate(self):
        """Return a new Brain with Gaussian noise applied to all weights."""
        new_Ws = []
        new_bs = []
        for W, b in zip(self._Ws, self._bs):
            W = W.copy()
            b = b.copy()
            mask_W = np.random.rand(*W.shape) < MUTATION_RATE
            mask_b = np.random.rand(*b.shape) < MUTATION_RATE
            W += mask_W * np.random.normal(0, MUTATION_STRENGTH, W.shape)
            b += mask_b * np.random.normal(0, MUTATION_STRENGTH, b.shape)
            np.clip(W, -3, 3, out=W)
            np.clip(b, -3, 3, out=b)
            new_Ws.append(W)
            new_bs.append(b)
        return Brain((new_Ws, new_bs))

    def clone(self):
        Ws, bs = self.get_weights()
        return Brain((Ws, bs))

    @property
    def layer_sizes(self):
        return self._sizes
