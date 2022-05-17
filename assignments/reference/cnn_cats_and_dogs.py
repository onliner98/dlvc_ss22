from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dlvc.models.pytorch import CnnClassifier

class CNN(nn.Module):
    def __init__(self, c_in) -> None:
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(c_in, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(576, 120)  # 400 ergibt sich bei input image 28x28
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_in, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, n_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding='same')
        self.block1 = ResBlock(64,64)
        self.downsample1 = nn.Conv2d(64, 128, 3, stride=2)
        self.block2 = ResBlock(128,128)
        self.downsample2 = nn.Conv2d(128, 256, 3, stride=2)
        self.block3 = ResBlock(256,256)
        self.fc = nn.Linear(12544, n_classes) # TODO change 400 to correct
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.downsample1(out)
        out = self.block2(out)
        out = self.downsample2(out)
        out = self.block3(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Constants
fp = 'C:/Users/admin/Desktop/10. Semester/Computer Vision/dlvc_ss22/assignments/reference/cifar10'
n_epochs = 100
batch_size=128
input_shape = (0, 3, 32, 32)
num_classes = 2
shuffle = True
lr = 0.01
wd = 0
op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw(),
])

train_ds = PetsDataset(fp, Subset.TRAINING)
valid_ds = PetsDataset(fp, Subset.VALIDATION)
test_ds = PetsDataset(fp, Subset.TEST)

train = BatchGenerator(train_ds, batch_size, shuffle, op)
valid = BatchGenerator(valid_ds, batch_size, shuffle, op)
test = BatchGenerator(test_ds, batch_size, shuffle, op)

#net = CNN(input_shape[1])
net = ResNet(input_shape[1], num_classes)  #input_shape[1] is the number of channels
# move to cuda if available
net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model = CnnClassifier(net, input_shape, num_classes, lr, wd)
acc = Accuracy()

for epoch in range(n_epochs):
    losses = []
    best_acc = 0
    # Training Loop
    for data in train:
        loss = model.train(data.data, data.label)
        losses.append(loss)
    # Validation Loop
    acc.reset()
    for data in valid:
        pred = model.predict(data.data)
        acc.update(pred, data.label)
    
    # Reporting
    losses = np.array(losses)

    print('epoch', epoch)
    print('train loss:', losses.mean(), '±', losses.std())
    print('val acc:', acc.accuracy())
    
    if acc.accuracy() > best_acc:
        torch.save(model, 'best_model.pth')
        best_acc = acc.accuracy()
# Aus der Aufgabenstelle ist nicht klar ersichtlich ob für test acc das best_model oder das model der letzten epoche verwendet werden soll
# Für das Modell der letzten epoche muss ledliglich die nächste zeile auskommentiert werden
model = torch.load('best_model.pth')
acc.reset()
for data in test:
    pred = model.predict(data.data)
    acc.update(pred, data.label)
print('--------------------')
print('val acc (best):', best_acc)
print('test acc:', acc.accuracy())