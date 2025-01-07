import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import sklearn

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

from sklearn.datasets import make_circles
n_samples = 1000

x, y = make_circles(
    n_samples = n_samples,
    noise = 0.03,
    random_state = seed,   
)

import pandas as pd

circles = pd.DataFrame({
    "x1": x[:, 0], 
    "x2": x[:, 1], 
    "label": y
})

# print(circles.head(10))

'''plt.scatter(
    x = x[:, 0],
    y = x[:, 1],
    c = y,
    cmap = plt.cm.RdYlBu
)
plt.show()'''

# print(x.shape)
# print(y.shape)

X = torch.from_numpy(x).to(device).type(torch.float32)
Y = torch.from_numpy(y).to(device).type(torch.float32)

train_percent = 0.7
train_split = int(train_percent * len(x))
validation_split = int((train_percent + (1 - train_percent)/2) * len(x))

train_x, train_y = X[:train_split], Y[:train_split]
validation_x, validation_y = X[train_split:validation_split], Y[train_split:validation_split]
test_x, test_y = X[validation_split:], Y[validation_split:]


class SinusActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)


class DivXActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        try:
            return 1/x
        except ZeroDivisionError:
            return 0


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1_1 = nn.Linear(
            in_features = 2,
            out_features = 8
        )

        # self.layer_1_2 = nn.Linear(
        #     in_features = 2,
        #     out_features = 8
        # )

        self.layer_1_3 = nn.Linear(
            in_features = 2,
            out_features = 8
        )

        # self.layer_1_4 = nn.Linear(
        #     in_features = 2,
        #     out_features = 8
        # )

        self.layer_2 = nn.Linear(
            in_features = 16,
            out_features = 1
        )

        '''self.two_linear_layers = nn.Sequential(
            nn.Linear(
                in_features = 2,
                out_features = 8
            ),
            
            nn.Linear(
                in_features = 8,
                out_features = 1
            )
        )''' # not gonna use this for now, better for simplifying more complex architectures

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sinus = SinusActivation()
        self.divx = DivXActivation()

    
    def forward(self, x):
        lyr_1_1 = self.layer_1_1(x)
        # lyr_1_2 = self.layer_1_2(x)
        lyr_1_3 = self.layer_1_3(x)
        # lyr_1_4 = self.layer_1_4(x)

        lyr_1_1 = self.leakyrelu(lyr_1_1)
        # lyr_1_2 = self.relu(lyr_1_2)
        lyr_1_3 = self.sinus(lyr_1_3)
        # lyr_1_4 = self.divx(lyr_1_4)

        lyr_2 = torch.cat((lyr_1_1, lyr_1_3), dim=1)

        out = self.layer_2(lyr_2)
        
        return out


model = CircleModelV1()
model.to(device)

loss_fn = nn.BCEWithLogitsLoss() # has softmax built in, is faster computationally
learning_rate = 0.01
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = learning_rate
)

def accuracy_fn(
    true_y,
    predicted_y
):
    correct = torch.eq(true_y, predicted_y).sum().item()
    acc = (correct/len(predicted_y))*100
    return acc

epochs = 300

epoch_count = []
train_accuracies = []
validation_accuracies = []
train_losses = []
validation_losses = []

for epoch in range(epochs):
    
    model.train()
    train_y_logits = model(train_x).squeeze()
    train_y_probabilities = torch.sigmoid(train_y_logits)
    train_y_labels = torch.round(train_y_probabilities)

    loss = loss_fn(train_y_logits, train_y)
    acc = accuracy_fn(true_y = train_y, predicted_y = train_y_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        validation_y_logits = model(validation_x).squeeze()
        validation_y_probabilities = torch.sigmoid(validation_y_logits)
        validation_y_labels = torch.round(validation_y_probabilities)

        validation_loss = loss_fn(validation_y_logits, validation_y)
        validation_acc = accuracy_fn(predicted_y = validation_y_labels, true_y = validation_y)
    
    epoch_count.append(epoch)
    train_accuracies.append(acc)
    validation_accuracies.append(validation_acc)
    train_losses.append(loss)
    validation_losses.append(validation_loss)

    print(f"E {epoch} - {(epoch/epochs)*100:.2f}% | T acc: {acc:.2f} | V acc: {validation_acc:.2f} | T loss: {loss:.5f} | V loss: {validation_loss:.5f}")


model.eval()
with torch.inference_mode():
    test_y_logits = model(test_x).squeeze()
    test_y_probabilities = torch.sigmoid(test_y_logits)
    test_y_labels = torch.round(test_y_probabilities)

    test_acc = accuracy_fn(predicted_y = test_y_labels, true_y = test_y)


total_params = sum(p.numel() for p in model.parameters())
print(f"\nTest accuracy: {test_acc:.5f}% | Params: {total_params} | Param/acc: {total_params/test_acc:.5f}")

epoch_count_cpu = epoch_count
train_accuracies_cpu = train_accuracies
validation_accuracies = validation_accuracies
train_losses = [value.item() for value in train_losses]
validation_losses = [value.item() for value in validation_losses]




fig, axs = plt.subplots(2, figsize=(10, 10))

axs[0].plot(epoch_count_cpu, train_accuracies_cpu, label="train acc")
axs[0].plot(epoch_count_cpu, validation_accuracies, label="valid acc")

axs[1].plot(epoch_count_cpu, train_losses, label="train loss")
axs[1].plot(epoch_count_cpu, validation_losses, label="valid loss")

axs[0].legend()
axs[1].legend()

axs[0].set_title("Accuracy")
axs[1].set_title("Loss")

plt.show()
