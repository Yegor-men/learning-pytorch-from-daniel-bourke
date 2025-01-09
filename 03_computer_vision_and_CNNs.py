import torch
from torch import nn
import torchmetrics
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}\n")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(), # converts to CHW tensor
    target_transform = None
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

class_names = train_data.classes

# print(train_data.class_to_idx)

# image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap = "gray")
# plt.title(f"{label} - {class_names[label]}")
# plt.show()

# fig = plt.figure(figsize = (9, 9))
# rows, cols = 4, 4
# for i in range(1, rows*cols + 1):
#     random_index = torch.randint(0, len(train_data), size = [1]).item()
#     img, label = train_data[random_index]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap = "gray")
#     plt.title(f"{class_names[label]} - {label}")
#     plt.axis(False)
# plt.show()

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

print(f"{len(train_dataloader)} batches of {BATCH_SIZE} in the training set")
print(f"{len(test_dataloader)} batches of {BATCH_SIZE} in the test set\n")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# random_index = torch.randint(0, len(train_features_batch), size = [1]).item()
# img, label = train_features_batch[random_index], train_labels_batch[random_index]
# plt.imshow(img.squeeze(), cmap = "gray")
# plt.title(class_names[label])
# plt.axis(False)
# print(f"\nImage shape: {img.shape}")
# print(f"Label: {label} | label size: {label.shape}")
# plt.show()

# ------------------------------------------------------------

class BaselineModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: int
    ):
        super().__init__()

        self.prelu = nn.PReLU()
        self.flatten = nn.Flatten()

        self.layer_1 = nn.Linear(
            in_features = input_size,
            out_features = hidden_units
        )

        self.layer_2 = nn.Linear(
            in_features = hidden_units,
            out_features = hidden_units
        )

        self.layer_3 = nn.Linear(
            in_features = hidden_units,
            out_features = output_size
        )

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.layer_1(x)
        x = self.prelu(x)
        x = self.layer_2(x)
        x = self.prelu(x)
        x = self.layer_3(x)
        
        return x

baseline_model = BaselineModel(
    input_size = 28*28,
    output_size = len(class_names),
    hidden_units = 256
).to(device)

# print(baseline_model.state_dict())


# ------------------------------------------------------------










# ------------------------------------------------------------

from timeit import default_timer as timer

model = baseline_model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)
acc_fn = torchmetrics.Accuracy(task = "multiclass", num_classes = len(class_names)).to(device)

from tqdm.auto import tqdm

EPOCHS = 3

start = timer()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch {epoch}\n-------")
    train_loss = 0

    for batch, (x_train, y_train) in enumerate(train_dataloader):
        model.train()
        x_train, y_train = x_train.to(device), y_train.to(device)

        y_train_pred = model(x_train)
        y_train_loss = loss_fn(y_train_pred, y_train)
        train_loss += y_train_loss

        optimizer.zero_grad()
        y_train_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Observed {batch * len(x_train)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    eval_loss, eval_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for x_eval, y_eval in test_dataloader:
            x_eval, y_eval = x_eval.to(device), y_eval.to(device)

            y_eval_pred = model(x_eval)

            eval_loss += loss_fn(y_eval_pred, y_eval)
            eval_acc += acc_fn(y_eval_pred.argmax(dim = 1), y_eval)
        
        eval_loss /= len(test_dataloader)
        eval_acc /= len(test_dataloader)
    
    print(f"tr loss: {train_loss:.3f} | ev loss: {eval_loss:.3f} | ev acc: {eval_acc*100:.3f}%")

end = timer()

print(f"\nTime elapsed: {(end - start):.3f}s")


