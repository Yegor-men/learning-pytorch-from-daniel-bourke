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