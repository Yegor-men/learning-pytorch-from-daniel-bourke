import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}\n")

num_classes = 4
num_features = 2
num_samples = 1000

x_raw_data, y_raw_data = make_blobs(
    n_samples = num_samples,
    n_features = num_features,
    centers = num_classes,
    cluster_std = 1.5,
    random_state = seed
)

x_blob = torch.from_numpy(x_raw_data).type(torch.float32).to(device)
y_blob = torch.from_numpy(y_raw_data).type(torch.float32).to(device)


train_split = int(0.7*len(x_blob))
eval_split = int(0.85 * len(x_blob))

x_train, y_train = x_blob[:train_split], y_blob[:train_split]
x_eval, y_eval = x_blob[train_split:eval_split], y_blob[train_split:eval_split]
x_test, y_test = x_blob[eval_split:], y_blob[eval_split:]

x_blob_cpu = x_blob.cpu()
y_blob_cpu = y_blob.cpu()

# plt.figure(figsize = (10, 7))
# plt.scatter(x_blob_cpu[:, 0], x_blob_cpu[:, 1], c = y_blob_cpu, cmap = plt.cm.RdYlBu)
# plt.show()


class BlobModel(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_layer_volume:int):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_volume = hidden_layer_volume

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()

        self.layer_1_1 = nn.Linear(
            in_features = self.input_size,
            out_features = self.hidden_layer_volume
        )

        self.layer_1_2 = nn.Linear(
            in_features = self.input_size,
            out_features = self.hidden_layer_volume
        )

        self.layer_1_3 = nn.Linear(
            in_features = self.input_size,
            out_features = self.hidden_layer_volume
        )

        self.layer_2_1 = nn.Linear(
            in_features = self.hidden_layer_volume*2,
            out_features = self.hidden_layer_volume*4
        )

        self.layer_2_2 = nn.Linear(
            in_features = self.hidden_layer_volume*2,
            out_features = self.hidden_layer_volume*4
        )

        self.layer_2_3 = nn.Linear(
            in_features = self.hidden_layer_volume*2,
            out_features = self.hidden_layer_volume*4
        )

        self.layer_3 = nn.Linear(
            in_features = self.hidden_layer_volume*12,
            out_features = self.output_size
        )
    
    def forward(self, x):
        x = x.type(torch.float)

        x1 = self.layer_1_1(x)
        x2 = self.layer_1_2(x)
        x3 = self.layer_1_3(x)

        x1, x2, x3 = self.prelu(x1), self.prelu(x2), self.prelu(x3)

        y1 = torch.cat((x1, x2), dim = 1)
        y2 = torch.cat((x2, x3), dim = 1)
        y3 = torch.cat((x3, x1), dim = 1)

        z1 = self.layer_2_1(y1)
        z2 = self.layer_2_2(y2)
        z3 = self.layer_2_3(y3)

        z1, z2, z3 = self.prelu(z1), self.prelu(z2), self.prelu(z3)

        o = self.layer_3(torch.cat((z1, z2, z3), dim = 1))

        return o
    
    def calculate_params(self):
        return sum(param.numel() for param in self.parameters())


model = BlobModel(
    input_size = num_features,
    output_size = num_classes,
    hidden_layer_volume = 1
)
model.to(device)

'''
def acc_fn(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc
'''

# could also use torchmetrics instead, but it does stuff in tensor form
from torchmetrics import Accuracy
torchmetric_acc = Accuracy(task = "multiclass", num_classes = num_classes).to(device)

EPOCHS = 100
loss_fn = nn.CrossEntropyLoss()
LEARNING_RATE = 0.01
optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr = LEARNING_RATE
)

epochs = []
tr_losses = []
ev_losses = []
tr_accs = []
ev_accs = []

for epoch in range(EPOCHS):
    
    model.train()

    y_train_logits = model(x_train)
    y_train_predictions = torch.softmax(y_train_logits, dim = 1).argmax(dim = 1)
    train_loss = loss_fn(y_train_logits, y_train.to(dtype=torch.long)) # as input wants int64
    train_acc = torchmetric_acc(y_train_predictions, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_eval_logits = model(x_eval)
        y_eval_predictions = torch.softmax(y_eval_logits, dim = 1).argmax(dim = 1)
        eval_loss = loss_fn(y_eval_logits, y_eval.to(dtype=torch.long))
        eval_acc = torchmetric_acc(y_eval_predictions, y_eval)
    
    epochs.append(epoch)
    tr_losses.append(train_loss)
    ev_losses.append(eval_loss)
    tr_accs.append(train_acc)
    ev_accs.append(eval_acc)    

    if (epoch % 10 == 0) or epoch == EPOCHS-1:
        print(f"E {epoch} - {(epoch/(EPOCHS-1))*100:.2f}% | tr loss: {train_loss:.5f} | ev loss = {eval_loss:.5f} | tr acc: {train_acc:.2f} | ev acc: {eval_acc:.2f}")

model.eval()
with torch.inference_mode():
    y_test_logits = model(x_test)
    y_test_predictions = torch.softmax(y_test_logits, dim = 1).argmax(dim = 1)
    test_acc = torchmetric_acc(y_test_predictions, y_test)

print(f"\nTest accuracy: {test_acc:.5f}% | Params: {model.calculate_params()} | Param/acc: {model.calculate_params()/test_acc:.5f}\n")

# Accuracy - decent for balanced samples - (tp+tn)/(tp+tn+fp+fn)
# Precision (less fp) - accuracy for positives - true positives/all positives - tp/(tp+fp)
# Recall (less fn) - true positives/all actual positives - tp/(tp+fn)
# F1-score - harmonic mean of precision and recall - 2*(precision*recall)/(precision+recall)
# Confusion matrix - summarizes predictions against actual outcomes
# Classification report - summarizes accuracy, precision, recall, f1-score

tr_losses_cpu = [x.item() for x in tr_losses]
ev_losses_cpu = [x.item() for x in ev_losses]
tr_accs_cpu = [x.item() for x in tr_accs]
ev_accs_cpu = [x.item() for x in ev_accs]

fig, axs = plt.subplots(2, figsize=(10, 10))

axs[0].plot(epochs, tr_losses_cpu, label = "tr loss")
axs[0].plot(epochs, ev_losses_cpu, label = "ev loss")
axs[1].plot(epochs, tr_accs_cpu, label = "tr acc")
axs[1].plot(epochs, ev_accs_cpu, label = "ev acc")

axs[0].legend()
axs[1].legend()

plt.show()