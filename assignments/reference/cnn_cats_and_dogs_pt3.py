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

def run_experiment(fp, net, name, n_epochs, batch_size, input_shape, num_classes, shuffle, lr, wd, op):
    train_ds = PetsDataset(fp, Subset.TRAINING)
    valid_ds = PetsDataset(fp, Subset.VALIDATION)
    test_ds = PetsDataset(fp, Subset.TEST)

    train = BatchGenerator(train_ds, batch_size, shuffle, op)
    valid = BatchGenerator(valid_ds, batch_size, shuffle, op)
    test = BatchGenerator(test_ds, batch_size, shuffle, op)

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
            torch.save(model, f'best_{name}.pth')
            best_acc = acc.accuracy()

    model = torch.load(f'best_{name}.pth')
    acc.reset()
    for data in test:
        pred = model.predict(data.data)
        acc.update(pred, data.label)
    print('--------------------')
    print('val acc (best):', best_acc)
    print('test acc:', acc.accuracy())
    return best_acc, acc.accuracy()





op_no_da = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw(),
])

op_da = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hflip(),
    ops.rcrop(32, 5, 'constant'),
    ops.hwc2chw(),
])

# try neither, only wd weak, only wd medium, only wd overreg, only da, both => was used to find best_config
# a config contains [wd, ops]
configs = [
    [0, op_no_da, 'cnn_neither'],
    [0.001, op_no_da, 'cnn_wd_low'],
    [0.1, op_no_da, 'cnn_wd_med'],
    [0.5, op_no_da, 'cnn_wd_over'],
    [0, op_da, 'cnn_da'],
    [0.1, op_da, 'cnn_both']
]

# Constants
fp = 'C:/Users/admin/Desktop/10. Semester/Computer Vision/dlvc_ss22/assignments/reference/cifar10'
n_epochs = 100
batch_size=128
input_shape = (3, 32, 32)
num_classes = 2
shuffle = True
lr = 0.01

net = CNN(input_shape[0])
# move to cuda if available
net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

op_hf = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hflip(),
    ops.hwc2chw(),
])
best_config = [[0.01, op_hf, 'cnn_low_wd_and_hf']]

results = []
for wd, op, name in best_config:
    best_val_acc, test_acc = run_experiment(fp, net, name, n_epochs, batch_size, input_shape, num_classes, shuffle, lr, wd, op)
    results.append([name, best_val_acc, test_acc])
    
print(results)