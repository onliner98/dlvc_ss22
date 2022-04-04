from abc import ABCMeta, abstractmethod

import numpy as np


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        self.right_prediction = 0
        self.predictions = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
            The predicted class label is the one with the highest probability.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"Shape mismatch. The two vector must have the same number of row")
        class_pred = np.argmax(prediction, axis=1)
        self.right_prediction += np.sum(class_pred == target)
        self.predictions += len(target)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        return f"accuracy: {self.accuracy()}"

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # See https://docs.python.org/3/library/operator.html for how these
        # operators are used to compare instances of the Accuracy class

        if not isinstance(other, Accuracy):
            raise TypeError(f'Invalid parameter type. Expected:{Accuracy}. Provided:{type(other)}.')
        return self.accuracy() < other.accuracy()

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not isinstance(other, Accuracy):
            raise TypeError(f'Invalid parameter type. Expected:{Accuracy}. Provided:{type(other)}.')
        return self.accuracy() > other.accuracy()

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        if self.predictions == 0:
            return 0
        return self.right_prediction / self.predictions
