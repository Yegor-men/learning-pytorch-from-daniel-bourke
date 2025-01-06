import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from savemodel import save_model
from loadmodel import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

start = 0
end = 1
step = 0.01

x = torch.arange(start, end, step, device = device).unsqueeze(dim = 1)
m = -0.3
c = 0.7
y = m * x + c

train_percent = 0.7
validation_percent = (1 - train_percent)/2

train_split = int(train_percent * len(x))
validation_split= int((train_percent + validation_percent) * len(x))

indices = torch.randperm(x.size(0))
x = x[indices]
y = y[indices]

train_x, train_y = x[:train_split],y[:train_split]
validation_x, validation_y = x[train_split:validation_split], y[train_split:validation_split]
test_x, test_y = x[validation_split:], y[validation_split:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                requires_grad = True,
                device = device,
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                1,
                requires_grad = True,
                device = device
            )
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

model = LinearRegressionModel()

epochs = 100
leaerning_rate = 0.01

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr = leaerning_rate)

epoch_count = []
train_loss_values = []
validation_loss_values = []

for epoch in range(epochs):
    model.train()
    train_y_pred = model(train_x)
    loss = loss_fn(train_y_pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        validation_y_pred = model(validation_x)
        validation_loss = loss_fn(validation_y_pred, validation_y)
    
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    validation_loss_values.append(validation_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | tain loss: {loss:.5f} | validation loss: {validation_loss:.5f}")

train_loss_values_cpu = [i.item() for i in train_loss_values]
validation_loss_values_cpu = [i.item() for i in validation_loss_values]

plt.plot(epoch_count, train_loss_values_cpu, label = "Training Loss")
plt.plot(epoch_count, validation_loss_values_cpu, label = "Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()