import numpy as np
import pickle
import os

from ..dataset import Sample, Subset, ClassificationDataset

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''    
        if not os.path.isdir(fdir):
            raise ValueError
        
        fdir_set = set(os.listdir(fdir))
        files = set(["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4","data_batch_5", "test_batch", "batches.meta"])
        if fdir_set.issubset(files):
            raise ValueError
        
        subsets = {
            Subset.TRAINING: np.array(["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]),
            Subset.VALIDATION: np.array(["data_batch_5"]),
            Subset.TEST: np.array(["test_batch"])
        }
        
        batches = subsets[subset]
        batch = self.unpickle(os.path.join(fdir,batches[0]))
        self.X, self.y = self.get_cats_and_dogs(fdir, batch)
        for batch_f in batches[1:]:
            batch = self.unpickle(os.path.join(fdir,batch_f))
            x_, y_ = self.get_cats_and_dogs(fdir, batch)
            self.X = np.concatenate((self.X, x_), axis=0)
            self.y = np.concatenate((self.y, y_), axis=0)
            
        self.len = self.X.shape[0]
        self.n_classes = np.unique(self.y).shape[0]
        self.X = self.X.reshape((self.len,3,32,32)).transpose(0,2,3,1) #array is red, blue, green => 3,32,32 => need for transpose
        self.X = np.flip(self.X, axis=3) #img is rgb but assignment wants bgr => flip b and r in axis 3
    
    def unpickle(self, file):
        '''
        Returns a dictionary containing data: 10000x3072 np array an labes: list of 10000 int from 0.9
        or a dictionary containing labelss_name: list of 10 strings if used on batches.meta
        '''
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def get_cats_and_dogs(self, fdir, batch):
        '''
        Returns data x_ and labels y_ from a dictionary filtered by cat and dog class
        '''
        meta = self.unpickle(os.path.join(fdir,"batches.meta"))
        cat = meta[b'label_names'].index(b'cat')
        dog = meta[b'label_names'].index(b'dog')
        
        x_ = np.array(batch[b'data'])
        y_ = np.array(batch[b'labels'])
        filter_mask = np.logical_or(y_ == cat, y_==dog)
        y_ = y_[filter_mask]
        y_[y_==cat]=0
        y_[y_==dog]=1
        return x_[filter_mask], y_

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return self.len

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''
        if idx < 0 or idx >=self.len:
            raise IndexError
        return Sample(idx, self.X[idx], self.y[idx])

        pass

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return self.n_classes
