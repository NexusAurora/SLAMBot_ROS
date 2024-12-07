import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear(nn.Module):
    
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        # Each element has its own weight for multiplication
        self.weights = nn.Parameter(torch.randn(input_dim))
        # Each element has its own bias for addition
        self.biases = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # Element-wise multiplication and addition
        return x * self.weights + self.biases

class Model(nn.Module):

    def __init__(self):
        
        super(Model, self).__init__()
        # Linear layer
        self.l1a = Linear(101)
        self.l1b = Linear(101)
        self.l1c = Linear(101)

        self.l6 = Linear(101)
        self.l7 = Linear(101)
        self.l8 = Linear(101)
        self.l9 = Linear(101)
        self.l10 = Linear(101)
        self.l11 = Linear(101)
        self.l12 = Linear(101)
        self.l13 = Linear(101)
        
        # Create a constant array (e.g., cosine values from 40 to 140 degrees)
        degrees = np.arange(40, 141)  # Create an array from 40 to 140
        radians = np.deg2rad(degrees)  # Convert degrees to radians
        cosine_values = np.cos(radians)  # Compute cosine
        
        # Convert the numpy array to a torch tensor and then to a Parameter, setting requires_grad to False
        self.cosine_constant = nn.Parameter(torch.tensor(cosine_values, dtype=torch.float32), requires_grad=False)

    def forward(self, x):

        #Each index is dependent on the previous and the next index 
        x_left_shift = torch.roll(x, 1, 1) 
        x_left_shift[:, -1] = 0

        x_right_shift = torch.roll(x, -1, 1)
        x_right_shift[:, 0] = 0

        x_pure_a = self.l1a(x)
        x_pure_b = self.l1b(x)

        x_front = self.l6(x_left_shift) + self.l7(x_right_shift) + x_pure_b
        x_front = F.gelu(self.l8(x_front))
        x_front = F.gelu(self.l9(x_front))
        x_front = self.l10(x_front)
        x_front = F.gelu(self.l11(x_front))
        x_front = self.l12(x_front)

        x_pure_c = self.l1c(x)
        x_angles = self.l13(self.cosine_constant*x_pure_c)

        return x + x_front + x_pure_a + x_angles
