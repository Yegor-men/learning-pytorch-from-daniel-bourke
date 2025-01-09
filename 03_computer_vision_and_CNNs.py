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

# class_names = train_data.classes

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

