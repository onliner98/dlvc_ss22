from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import dlvc.ops as ops
import numpy as np

# 1. Load the training, validation, and test sets as individual PetsDatasets.

fp = '/mnt/1028D91228D8F7A4/Python Project/PycharmProjects/DeepLearning/assignments/src/cifar10'

print("Load data")
train_ds = PetsDataset(fp, Subset.TRAINING)
valid_ds = PetsDataset(fp, Subset.VALIDATION)
test_ds = PetsDataset(fp, Subset.TEST)
print("Data Loaded")

# 2. Create a BatchGenerator for each one. Traditional classifiers don't usually train in batches so you can set the
# minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier
# that does require multiple batches.
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
])


train = BatchGenerator(train_ds, len(train_ds), False, op)
print(f"number of training batches when batch size = len(dataset): {len(train)}")
print("--------------------")
train = BatchGenerator(train_ds, 500, False, op)
print(f"number of training batches when batch size = 500: {len(train)}")
print("Data and label shapes")



for i in train:
    print(f"Data shape: {i.data.shape}; Data Type: {i.data.dtype}")
    print(f"Label shape: {i.label.shape}; Label Type: {i.label.dtype}")
    print("\n")

print("-----------")
print("First training batch without shiffling")
print(next(iter(train)).data[0][0:5])
print(next(iter(train)).label[0:5])

print("-----------")
print("First training batch with shiffling")
train = BatchGenerator(train_ds, 500, True, op)
print(next(iter(train)).data[0][0:5])
print(next(iter(train)).label[0:5])

