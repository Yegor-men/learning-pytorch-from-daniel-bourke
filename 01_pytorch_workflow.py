import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import random

seed = 42

# random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.no_grad() defines boundaries for code where gradients are not created
with torch.no_grad():
    # Using linear regression to create a straight line to have known parameters
    m = 0.7
    c = 0.3

    start, end, step = 0, 1, 0.01
    X = torch.arange(start, end, step).unsqueeze(dim = 1) # makes it a (50, 1), dim = 2
    y = m * X + c # y inherits the shape of X, because * is short notation of element wise multiplication

    X, y = X.to(device), y.to(device) # sending to cuda

    # Shuffling the data
    indices = torch.randperm(X.size(0)) # random permutation
    X_shuffled = X[indices].clone() # X[indices] is a view, .clone makes a copy
    y_shuffled = y[indices].clone()
    del X, y # deleting X and y to free up memory

    # print(f"{X_shuffled[:10]}, {y_shuffled[:10]}")
    # print(f"{len(X)}, {len(y)}")

    # a validation set is sometimes useful for checking accuracy during training
    train_split = int(0.7 * len(X_shuffled))
    validation_split = int(0.85 * len(X_shuffled))

    train_x, train_y = X_shuffled[:train_split], y_shuffled[:train_split]
    validation_x, validation_y = X_shuffled[train_split:validation_split], y_shuffled[train_split:validation_split]
    test_x, test_y = X_shuffled[validation_split:], y_shuffled[validation_split:]

    torch.cuda.empty_cache() # free up memory


def plot_predictions(train_data: torch.Tensor = train_x.cpu(), 
                     train_labels: torch.Tensor = train_y.cpu(), 
                     validation_data: torch.Tensor = validation_x.cpu(), 
                     validation_labels: torch.Tensor = validation_y.cpu(), 
                     test_data: torch.Tensor = test_x.cpu(), 
                     test_labels: torch.Tensor = test_y.cpu(), 
                     predictions: torch.Tensor = None):
    # matplotlib cannot work with gpu tensors, must use .cpu() to make copy
    
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_labels, c = "r", s = 4, label = "Training data")
    plt.scatter(validation_data, validation_labels, c = "g", s = 4, label = "Validation data")
    plt.scatter(test_data, test_labels, c = "b", s = 4, label = "Test data")

    if predictions is not None:
        predictions = predictions.to('cpu', non_blocking=True)
        plt.scatter(test_data, predictions, c = "y", s = 4, label = "Predictions")
    
    plt.legend(prop = {"size": 14})
    plt.show()


# plot_predictions()

# Building a 1 neuron linear regression model

class LinearRegressionModel(nn.Module): # subclass is nn.module
    def __init__(self):
        super().__init__() # calls init from parent class
        self.weight = nn.Parameter(torch.randn(1, 
                                                requires_grad = True,
                                                dtype = torch.float32,
                                                device = device))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad = True,
                                             dtype = torch.float32,
                                             device = device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -> means that it returns a torch tensor
        return self.weight * x + self.bias

model_0 = LinearRegressionModel().to(device)
print(model_0.state_dict())

# making predictions with torch.inference_mode()


# picking a loss function and optimizer (MAE and stochastic gradient descent)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr = 0.01)

# training loop and testing loop
epochs = 500

epoch_count = []
train_loss_values = []
validation_loss_values = []

for epoch in range(epochs):
    model_0.train() # turns on the training mode for gradients, 
    
    pred_y = model_0(train_x) # sets the predictions
    loss = loss_fn(pred_y, train_y) # first input then target
    # print(f"Loss: {loss}")
    optimizer.zero_grad() # clears the gradients so that old grads dont affect current
    loss.backward() # does the backprop
    optimizer.step() # makes the optimizer step
    
    model_0.eval() # turns off trainig mode for gradients
    with torch.inference_mode():
        validation_predictions = model_0(validation_x)
        validation_loss = loss_fn(validation_predictions, validation_y)

    if epoch % 1 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss)
        validation_loss_values.append(validation_loss)
    
        print(f"E-{epoch}, {(epoch / epochs)*100:.2f}% | Loss: {loss:.6f} | Validation loss: {validation_loss:.6f} | Difference: {loss - validation_loss:.6f}")


with torch.inference_mode(): # inference mode turns off gradient tracking
    test_predictions = model_0(test_x)
# plot_predictions(predictions = test_predictions)


train_loss_values_cpu = [loss.item() for loss in train_loss_values]
validation_loss_values_cpu = [loss.item() for loss in validation_loss_values]

plt.plot(epoch_count, train_loss_values_cpu, label = "Training Loss")
plt.plot(epoch_count, validation_loss_values_cpu, label = "Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# plt.show()

# Saving models
# torch.save() as pickle
# torch.load()
# torch.nn.Module.load_state_dict()

from savemodel import save_model
save_model(name = "01_pytorch_workflow.py", path = "models", model = model_0)

# uses statedict so it needs to be instantiated to the same class model
from loadmodel import load_model
loaded_model_0 = LinearRegressionModel()
load_model(instance = loaded_model_0, path = "models/01_pytorch_workflow_model_0.pth")

loaded_model_0.eval()
with torch.inference_mode():
    test_predictions = loaded_model_0(test_x)

plot_predictions(predictions = test_predictions)