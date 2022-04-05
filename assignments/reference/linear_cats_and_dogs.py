from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch


# TODO: Define the network architecture of your linear classifier.
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # define network layer
        self.layer = torch.nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x):
        return self.layer(x)


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
])


def train_model(linear_classifier, criterion, optimizer, epochs, train_data, valid_data):
    acc = Accuracy()
    print("Train the network")
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        acc.reset()
        for data in train_data:
            # get the inputs and the labels
            inputs = data.data
            labels = data.label

            # convert the np.array in tensor
            t_inputs = torch.tensor(inputs)
            t_labels = torch.tensor(labels)

            # zero the parameter gradients, for every batch I must compute the gradient again
            optimizer.zero_grad()

            # forward step
            output = linear_classifier.forward(t_inputs)
            loss = criterion(output, t_labels)
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameter

            # print statistics
            running_loss += loss.item()
            acc.update(linear_classifier.forward(torch.tensor(valid_data.data)).detach().numpy(), valid_data.label)
            print(f"epoch {epoch + 1} \ntrain loss: {running_loss}\nval accuracy: {acc.accuracy()}")

            # update the values
            running_loss = 0.0
            if acc.accuracy() >= best_acc:
                best_acc = acc.accuracy()

            acc.reset()
    print("Finished Training")
    return linear_classifier, best_acc


def main():
    fp = '/mnt/1028D91228D8F7A4/Python Project/PycharmProjects/DeepLearning/assignments/reference/cifar10'

    print("Load data")
    train_ds = PetsDataset(fp, Subset.TRAINING)
    valid_ds = PetsDataset(fp, Subset.VALIDATION)
    test_ds = PetsDataset(fp, Subset.TEST)
    print("Data Loaded")

    print("Creating Batch Generator")
    train = BatchGenerator(train_ds, len(train_ds), False, op)
    valid = next(iter(BatchGenerator(valid_ds, len(valid_ds), False, op)))
    test = next(iter(BatchGenerator(test_ds, len(test_ds), False, op)))
    print("Batch Generator created")

    #define general parameters
    in_features = 3072  # size of the vector in input
    epochs = 100

    #Test 1
    print("Create Linear Classifier, Loss Function and Optimizer")
    lc = LinearClassifier(in_features, train_ds.num_classes())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lc.parameters(), lr=0.001, momentum=0.9)

    lc_test_1, best_acc_test1 = train_model(lc, criterion, optimizer, epochs, train, valid)

    #Test 2, change the optimizer
    print("------------------------------------")
    lc = LinearClassifier(in_features, train_ds.num_classes())
    optimizer = torch.optim.Adam(lc.parameters(), lr=0.001)
    lc_test_2, best_acc_test2 = train_model(lc, criterion, optimizer, epochs, train, valid)

    #find the best model
    if best_acc_test1 > best_acc_test2:
        lc = lc_test_1
        best_acc = best_acc_test1
    else:
        lc = lc_test_2
        best_acc = best_acc_test2

    print("--------------------")
    print(f"val accuracy (best): {best_acc}")

    # compute the test accuracy
    test_acc = Accuracy()
    test_acc.update(lc.forward(torch.tensor(test.data)).detach().numpy(), test.label)
    print(f"test accuracy: {test_acc.accuracy()}")


main()