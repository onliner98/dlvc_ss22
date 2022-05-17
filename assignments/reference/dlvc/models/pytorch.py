import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..model import Model

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''
        self.net = net
        self.in_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd
        self.is_cuda = list(net.parameters())[0].is_cuda
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd, nesterov=True)
        
        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''
        return self.in_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''
        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        # Raise Errors
        if data.dtype != np.float32 or labels.dtype !=int:
            raise TypeError
        if (not self.check_shape(data.shape)):
            raise ValueError
        if (labels<0).any() or (labels>self.num_classes - 1).any():
            raise ValueError
        try:
            # Make sure to set the network to train() mode
            self.net.train(True)

            # See above comments on CPU/GPU
            if self.is_cuda:
                data = torch.from_numpy(data).cuda()
                labels = torch.from_numpy(labels).cuda()
            else:
                data = torch.from_numpy(data).cpu()
                labels = torch.from_numpy(labels).cpu()

            # Training
            self.optimizer.zero_grad()   # zero the gradient buffers
            output = self.net(data)
            loss = self.criterion(output, labels.to(torch.long)) #cast labels to long so the CE works (CE throws exception with int)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except:
            raise RuntimeError

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()). # output_shape()=self.num_classes, with n you probably mean m?
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values. 
        Raises RuntimeError on other errors.
        '''
        
        # Raise Errors
        if not self.check_shape(data.shape):
            raise ValueError
        if data.dtype != np.float32:
            raise TypeError
        try:
            # Make sure to set the network to eval() mode
            self.net.eval()
            # See above comments on CPU/GPU
            if self.is_cuda:
                data = torch.from_numpy(data).cuda()
            else:
                data = torch.from_numpy(data).cpu()
                
            output = nn.Softmax(dim=1)(self.net(data))
            return output.detach().cpu().numpy()
        except:
            raise RuntimeError
            
    def check_shape(self, shape):
        '''Checks whether a given shape complys to the input shape of the network'''
        correct_shape = False
        if(len(shape)==4):
            _, c, h, w = shape
            _, C, H, W = self.in_shape
            if c==C and h==H and w==W:
                correct_shape=True
        return correct_shape