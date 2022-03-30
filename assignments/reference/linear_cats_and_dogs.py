from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch
import torch.nn.functional as F


# TODO: Define the network architecture of your linear classifier.
class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
    super(LinearClassifier, self).__init__()
    # TODO: define network layer(s)
    

  def forward(self, x):
    # TODO: Implement the forward pass.
    return x

# TODO: Create a 'BatchGenerator' for training, validation and test datasets.
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])


# TODO: Create the LinearClassifier, loss function and optimizer. 


'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your findings in the report.
'''


