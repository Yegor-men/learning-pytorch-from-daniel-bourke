import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"{torch.__version__}\n")

# Torch tensors: https://pytorch.org/docs/stable/tensors.html

# Scalar tensor
scalar = torch.tensor(7) # a scalar (0 dimension tensor) of value 7
print(f"Scalar: {repr(scalar)}")
print(f"Dimensions: {scalar.ndim}") # ndim is the number of dimensions it has
print(f"Value: {scalar.item()}") #  returns the value of the scalar tensor

print()

# Vector tensor
vector = torch.tensor([7, 8, 9]) # a vector (1 dimension tensor) of value 6,9
print(f"Vector: {vector}")
print(f"Dimensions: {vector.ndim}")
print(f"Shape: {vector.shape}") # returns the number of elemnts it has
print(f"Value: {vector.tolist()}") # non0 dimensional analogue of .item()

print()

'''
In mathematics and physics it is convention to:
    use lowercase for scalars and vectors
    use uppercase for MATRIX and TENSOR
'''

# MATRIX tensor
MATRIX = torch.tensor([[1, 2],
                       [3, 4]
                       ])
print(f"MATRIX: {MATRIX}")
print(f"Dimensions: {MATRIX.ndim}")
print(f"Shape: {MATRIX.shape}") # another way to think of this is as 2 arrays of 2 elements
print(f"Value: {MATRIX.tolist()}")

print()

# TENSOR tensor
TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]],
                       [[13, 14, 15], [16, 17, 18]]
                       ])
print(f"TENSOR: {TENSOR}")
print(f"Dimensions: {TENSOR.ndim}")
print(f"Shape: {TENSOR.shape}") # 3 arrays of 2 arrays of 3 elements
print(f"Value: {TENSOR.tolist()}")
print(f"First element: {TENSOR[0]}")

print()

'''
for the sake of now on:
    scalars are prepended by s_
    vectors are prepended by v_
    matrices are prepended by m_
    tensors are prepended by t_

thus drastically reducing complexity and increasing comprehension
'''

# Random tensors: https://pytorch.org/docs/main/generated/torch.rand.html
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# creating a random tensor of size (3, 4)
m_random_tensor = torch.rand(3, 4)
print(f"Random tensor: {m_random_tensor}")
print(f"Shape: {m_random_tensor.shape}") # as expected, (3, 4)

print()

# creating a random tensor of size similar to that of an image
# size should be (number of color channels, width, height)
# color channels go first for intuitive readability, as it will be 3 arrays of 224x224
t_random_image_tensor = torch.rand(size = (3, 224, 224))
print(f"Tensor for images of 224w x 224h and RGB channels")
print(f"Shape: {t_random_image_tensor.shape}, Dimensions: {t_random_image_tensor.ndim}")

print()

# Zero tensor
m_zeros = torch.zeros(size = (3, 4))
print(f"Zeros tensor: {m_zeros}")

print()

# tensors can also be multiplied by scalars and other tensors
print(f"Zeros by random tensor: {m_zeros * m_random_tensor}") # elementwise multiplication
print(f"Random tensor by scalar (7): {m_random_tensor * scalar}")

print()

# Ones tensor
m_ones = torch.ones(size = (3, 4))
print(f"Ones tensor: {m_ones}")
print(f"Data type of ones tensor: {m_ones.dtype}")

print()

# Torch.arange allows for creating ranges: https://pytorch.org/docs/stable/generated/torch.arange.html
v_arange = torch.arange(start = 0, end = 1000, step = 77) # end value is excluded
print(f"Arange: {v_arange}")
print(f"Shape: {v_arange.shape}, Dimensions: {v_arange.ndim}")

print()

# Creating like tensors, the same shape as another tensor
v_zeros = torch.zeros_like(v_arange)
print(f"Zeros like arange: {v_zeros}")
print(f"Shape: {v_zeros.shape}, Dimensions: {v_zeros.ndim}")
# also allows for ones_like and rand_like

print()

# Tensor datatypes: https://pytorch.org/docs/stable/tensors.html
float_32_tensor = torch.tensor([1.0, 2.0, 3.0], 
                                dtype = None, # default float32
                                device = None,
                                requires_grad = False)
# what datatype the tensor is
# device can be cpu or cuda
# requires_grad is for tracking gradients
print(f"Float 32 tensor: {float_32_tensor}")
print(f"Data type of float 32 tensor: {float_32_tensor.dtype}")

print()

float_16_tensor = float_32_tensor.type(torch.float16)
print(f"Float 16 tensor: {float_16_tensor}")
print(f"Data type of float 16 tensor: {float_16_tensor.dtype}")

print()

# Getting info about a tensor:
cuda0 = torch.device('cuda:0')
some_tensor = torch.rand(size = (3, 4),
                        dtype = None,
                        device = cuda0,
                        requires_grad = False)
print(f"some_tensor: {some_tensor}")
print(f"Shape: {some_tensor.shape}") # attribute
print(f"Dimensions : {some_tensor.ndim}")
print(f"Size: {some_tensor.size()}") # function
print(f"Device: {some_tensor.device}")
print(f"Type: {some_tensor.dtype}")
print(f"On cuda?: {some_tensor.is_cuda}")

print()

# Manipulating tensors (tensor operations)
# addition, subtraction, multiplication (element wise), division, matrix multiplication

tensor = torch.tensor([1, 2, 3])
print(f"Tensor: {tensor}")
print(f"Tensor +10: {tensor + 10}")
print(f"Tensor -10: {tensor - 10}")
print(f"Tensor *10: {tensor * 10}")
print(f"Tensor /10: {tensor / 10}")
print(f"Tensor * Tensor: {tensor * tensor}")
print(f"Tensor / Tensor: {tensor / tensor}")

print(f"Tensor +10: {torch.add(tensor, 10)}")
print(f"Tensor -10: {torch.sub(tensor, 10)}")
print(f"Tensor *10: {torch.mul(tensor, 10)}")
print(f"Tensor /10: {torch.div(tensor, 10)}")

print(f"Matmul of 1D tensor = as if 2nd transposed:{torch.matmul(tensor, tensor)}")

print()

# Matrix multiplication (must be atleast 2 dimensions to take transpose)
tensor = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
print(f"Tensor: {tensor}, shape:{tensor.shape}")
tensor_t = tensor.T # switches axes for x and y of the matrix, T for transpose
print(f"Transmuted tensor: {tensor.T}")
print(f"Tensor dot Tensor.T: {torch.matmul(tensor, tensor.T)}")
print(f"Shape:{torch.mm(tensor, tensor.T).shape}") # torch.mm is an alias for torch.matmul
print(f"Tensor.T dot Tensor: {tensor.T @ tensor}") # @ is a short notation for matmul
print(f"Shape:{torch.matmul(tensor.T, tensor).shape}")

print()

# Tensor aggregation: min, max, mean, sum, etc
# torch.randn is normal distribution w/ mean of 0 and stdev of 1
# torch.rand is uniform distribution between 0 and 1

# time stopped at 2.48.45