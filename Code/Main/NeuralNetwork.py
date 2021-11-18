# Ben Lehrburger
# Philanthropy in higher education classifier

# Define the parameters of a neural network object

# ***DEPENDENCIES***
import torch.nn as nn

# Wrap a neural network object
class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # Define the network's architecture
        self.linear_relu_stack = nn.Sequential(

            # Input layer has 21 nodes for the 21 data features
            nn.Linear(21, 256),

            # Nonlinear ReLU layer
            nn.ReLU(),

            # Intermediate linear hidden layer
            nn.Linear(256, 512),

            # Nonlinear ReLU layer
            nn.ReLU(),

            # Output layer has 12 nodes for the 12 classes
            nn.Linear(512, 12),

        )

    # Check if a network prediction matches its label in training
    def is_correct(self, L, P):
        if L == P:
            return True

    # Forward-propagate a data sample through the network to make prediction
    def forward(self, x):
        return self.linear_relu_stack(x)

    # Custom nonlinear function that reduces 12 class output to 5 classes
    def nonlinear_bracket(self, L, P):

        class_label = None

        # Class 0: $0
        if L == 0:
            class_label = 0
            if P == 0:
                return True, class_label

        # Class 4: $25K+
        elif L == 11:
            class_label = 4
            if P == 11:
                return True, class_label

        # Class 1: $1-$999
        elif L == 1 or L == 2 or L == 3 or L == 4 or L == 5 or L == 6:
            class_label = 1
            if P == 1 or P == 2 or P == 3 or P == 4 or P == 5 or P == 6:
                return True, class_label

        # Class 2: $1K-$4.9K
        elif L == 7 or L == 8:
            class_label = 2
            if P == 7 or P == 8:
                return True, class_label

        # Class 3: $5K-$24.9K
        elif L == 9 or L == 10:
            class_label = 3
            if P == 9 or P == 10:
                return True, class_label

        return False, class_label

    # Employ the nonlinear output function in prediction-making
    def classify(self, prediction):

        # Class 0: $0
        if prediction == 0:
            return 0

        # Class 4: $25K+
        elif prediction == 11:
            return 4

        # Class 1: $1-$999
        elif prediction == 1 or prediction == 2 or prediction == 3 or prediction == 4 or prediction == 5 or prediction == 6:
            return 1

        # Class 2: $1K-$4.9K
        elif prediction == 7 or prediction == 8:
            return 2

        # Class 3: $5K-$24.9K
        elif prediction == 9 or prediction == 10:
            return 3
