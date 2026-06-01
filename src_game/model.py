import os
import numpy as np

class MNISTClassifier:
    def __init__(self, weights_path='assets/weights.npz'):
        self.weights_path = weights_path
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None
        self.is_loaded = False
        
        self.load_weights()
        
    def load_weights(self):
        """Loads weights from the npz file. Handles FileNotFoundError."""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Model weights file not found at '{self.weights_path}'. "
                "Please run 'train_model.py' to train the model and generate the weights first."
            )
        try:
            data = np.load(self.weights_path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.is_loaded = True
        except Exception as e:
            raise IOError(f"Failed to load weights from file: {e}")

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Subtract max for numerical stability (prevent overflow)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Runs the forward pass for the 3-layer MLP.
        x: Flattened input vector of shape (784,) or batch (N, 784)
        """
        if not self.is_loaded:
            raise RuntimeError("Model weights are not loaded.")
            
        # Layer 1: fc1 (784 -> 128) + ReLU
        # input x shape: (784,) -> reshape to (1, 784) for matrix multiplication
        # W1 shape: (128, 784), b1 shape: (128,)
        # output z1 = x . W1^T + b1
        z1 = np.dot(x, self.W1.T) + self.b1
        a1 = self.relu(z1)
        
        # Layer 2: fc2 (128 -> 64) + ReLU
        # W2 shape: (64, 128), b2 shape: (64,)
        z2 = np.dot(a1, self.W2.T) + self.b2
        a2 = self.relu(z2)
        
        # Layer 3: fc3 (64 -> 10) + Softmax
        # W3 shape: (10, 64), b3 shape: (10,)
        z3 = np.dot(a2, self.W3.T) + self.b3
        probabilities = self.softmax(z3)
        
        return probabilities

    def predict(self, image_28x28):
        """
        Predicts the digit from a 28x28 image.
        image_28x28: numpy array of shape (28, 28) with values in range [0, 1] or [0, 255]
        Returns:
            predicted_digit: int (0-9)
            probabilities: list of 10 floats (confidence levels)
        """
        # Preprocessing: flatten to shape (784,)
        x = image_28x28.flatten()
        
        # Normalize if not normalized (assuming range [0, 255])
        if x.max() > 1.0:
            x = x / 255.0
            
        # MNIST standardization using train mean=0.1307, std=0.3081
        x = (x - 0.1307) / 0.3081
        
        # Forward pass
        probs = self.forward(x)
        
        # If input was 1D vector (784,), probs will be (10,) or (1, 10)
        probs = probs.flatten()
        
        predicted_digit = int(np.argmax(probs))
        return predicted_digit, probs.tolist()
