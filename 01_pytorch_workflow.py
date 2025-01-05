import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Using linear regression to create a straight line to have known parameters
# y = mx + c
m = 0.7
c = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim = 1) # makes it a (50, 1), dim = 2
y = m * X + c

print(f"{X[:10]}, {y[:10]}")
print(f"{len(X)}, {len(y)}")

