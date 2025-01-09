'''
addition, subtraction, multiplication, division, matmul, transpose
are done in short notation
+   -   *    /    @   .T

min, max, mean, sum, argmin, argmax (and view)
are considered attributes
x.min()   x.max()   x.mean()   x.sum()   x.argmin()   x.argmax()   x.view(dim1, dim2)

reshape, view, stack, squeeze, unsqueeze, permute
are considered functions
torch.reshape(x, (3, 3)), torch.stack([x, x], dim = 1), torch.squeeze(x), torch.unsqueeze(x, 0), torch.permute(x, (1, 0))
'''

'''
If it is necessary to remember what dimension tensor it is, tensors are named as follows:
x_dim0
x_dim1
x_dim2
etc
'''

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from savemodel import save_model
# save_model(name = , path = )
from loadmodel import load_model
# load_model(name = , path = )
'''

'''
from timeit import default_timer as timer

start = timer()
end = timer()
total_time = end - start
print(f"Total time: {total_time:.5f}s")

'''