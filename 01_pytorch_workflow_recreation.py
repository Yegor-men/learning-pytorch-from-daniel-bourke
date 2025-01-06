import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from savemodel import save_model
from loadmodel import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

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
noise = torch.randn_like(x) * 0.01
y = m * x + c + noise

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
        
        self.linear_layer = nn.Linear(
            in_features = 1,
            out_features = 1,
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model = LinearRegressionModel()
model.to(device)

epochs = 150
learning_rate = 0.01

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epoch_count = []
train_loss_values = []
validation_loss_values = []

prev_difference = 0

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
        loss_difference = loss - validation_loss
        accel = loss_difference - prev_difference
        prev_difference = loss_difference
    
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    validation_loss_values.append(validation_loss)

    if epoch % 1 == 0:
        print(f"E{epoch} - {(epoch/epochs)*100:.2f}% | t loss: {loss:.5f} | v loss: {validation_loss:.5f} | Difference score: {torch.log(loss_difference):.5f} | Accel = {accel:.5f}")

train_loss_values_cpu = [i.item() for i in train_loss_values]
validation_loss_values_cpu = [i.item() for i in validation_loss_values]

plt.figure(figsize = (10, 7))
plt.plot(epoch_count, train_loss_values_cpu, label = "Training Loss")
plt.plot(epoch_count, validation_loss_values_cpu, label = "Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

X = test_x.cpu()
Y = test_y.cpu()
plt.figure(figsize = (10, 7))
plt.scatter(X, Y, c = "b",s = 4, label = "Test Data")
with torch.inference_mode():
    test_y_pred = model(test_x)
    test_loss = loss_fn(test_y_pred, test_y)
test_loss_cpu = test_loss.item()
test_y_pred_cpu = test_y_pred.cpu()
plt.plot(X, test_y_pred_cpu, c = "r", label = f"Loss: {test_loss_cpu:.5f}")
plt.title("Model predictions to test data")
plt.legend()
plt.show()