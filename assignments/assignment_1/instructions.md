# Deep Learning for Visual Computing - Assignment 1

The first assignment allows you to become familiar with basic dataset handling, image processing, and machine learning.

This text or the reference code might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@cvl.tuwien.ac.at). Students who find and report such errors will get extra points.

## Part 0

This part is about setting up the environment for developing. 

All assignments will be implemented in Python 3 and [PyTorch](https://pytorch.org/). So first make sure Python 3.6 or newer is installed on your computer as that's the minimal requirement of the most recent PyTorch version. If not, [download](https://www.python.org/downloads/) and install a recent version.

Then setup, create, and enable a [virtualenv](https://virtualenv.pypa.io/en/stable/). This facilitates package installation and ensures that these packages don't interfere with other Python code you might already have. Once done, make sure `$ python --version` returns something like `python 3.7.0`. Finally, install the core packages we'll need:

    pip install numpy opencv-python

The PyTorch setup varies a bit depending on the OS, see [here](https://pytorch.org/). Use a version with CUDA only if you have an Nvidia GPU. In any case, ensure to install PyTorch version 1.8.2. This is the version we will use for testing all assignments and if they fail due to version issues, you'll get significant point deductions. Confirm this via:

    python -c "import torch; print(torch.__version__)"

## Part 1

Download the reference code from [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/tree/main/assignments/reference), making sure that the file structure is preserved, and rename the root folder to something other than `reference`. Read the code and make sure that you understand what the individual classes and methods are doing.

[Download](https://www.cs.toronto.edu/~kriz/cifar.html) and extract the *Python* version of the CIFAR10 dataset somewhere *outside* the code folder. Read the website to understand which classes there are and how the data are structured and read.

Then implement the `PetsDataset` (`datasets/pets.py`). Some parts of this assignment will be tested automatically so make sure to follow the instructions in the code exactly (note, for example the specified data types). Make sure the following applies. If not, you made a mistake somewhere:

* Number of samples in the individual datasets: 7959 (training), 2041 (validation), 2000 (test).
* Total number of cat and dog samples: 6000 per class
* Image shape: always `(32, 32, 3)`, image type: always `uint8`
* Labels of first 10 training samples: `0 0 0 0 1 0 0 0 0 1`
* Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (`cv2.imshow`, `cv2.imwrite`).

Do not modify, add or delete any other files.

## Part 2

Make sure you have the most recent [reference code](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/tree/main/assignments/reference). Next to updated code we will occasionally add clarifications or correct errors. You can check the individual commits to see what has changed.

In this part we will implement common functionality for classifier training. As we'll see in the lecture, training and testing is almost always done in mini-batches, with each being a small part of the whole data. To do so, finish the `BatchGenerator` class in `batches.py`. Make sure to read the comments and implement type and value checks accordingly.

The `BatchGenerator`'s constructor has an optional `op` argument that is a function. If this argument is given, the generator will apply this function to the data of every sample before adding it to a batch. This is a flexible mechanism that will later allow us to implement data augmentation. For now we'll use it to transform the data to the form expected by classifiers such as KNN. For this we need to convert the images to float vectors, as covered in the lecture. To do so, implement the `type_cast`, `vectorize`, `add` and `mul` functions inside `ops.py`. These are functions that return other functions. See the `chain` function, which is already implemented for reference. That function allows for chaining other operations together like so:

```python
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
```

We will use the `add()` and `mul()` operations for basic input normalization. The above arguments will scale the vector entries to the interval `[-1, 1]`.

To test the batch generator make sure the following applies (the following list assumes that the above op chain is used):

* The number of training batches is `1` if the batch size is set to the number of samples in the dataset
* The number of training batches is `16` if the batch size is set to 500
* The data and label shapes are `(500, 3072)` and `(500,)`, respectively, unless for the last batch which will be smaller
* The data type is always `np.float32` and the label type is `np.int64`
* The first sample of the first training batch returned *without shuffling* has label `0` and data `[-0.09019608 -0.01960784 -0.01960784 -0.28627452 -0.20784315 ...]`.
* The first sample of the first training batch returned *with shuffling* must be random.

Finally we will use accuracy as the performance measure for our classifiers. See the lecture slides for how this measure is defined and implement the `Accuracy` class in `test.py` accordingly. This class supports batch-wise updates which will be handy in the future.

## Part 3

In this part we will implement a simple classifier and test all of the code implemented so far. At this point the classifier will not be a deep neural network. Instead we will implement a linear classifier with PyTorch that serves as a baseline for future results. Get the `linear_cats_and_dogs.py` script from the latest version of the [reference code](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/tree/main/assignments/reference) and implement the missing functionalities (see TODOs). 

The `linear_cats_and_dogs.py` script should do the following, in this order: 
1. Implement the network architecture of the linear classifier `LinearClassifier` (use a single `torch.nn.Linear` layer).
2. Load the training, validation, and test sets as individual `PetsDataset`s.
3. Create a `BatchGenerator` for each dataset using the input transformation chain `op`. Set the minibatch size equal to the number of dataset samples to get a single large batch.
4. Select optimization critereon and optimizer. Use `torch.nn.CrossEntropyLoss()` as optimization critereon and pick an optimizer from `torch.optim`. Popular choices are `torch.optim.SGD` or `torch.optim.Adam`, however, you are free to experiment with other optimizers (and their parameters).   
5. Train your `LinearClassifier` for at least 100 epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. This should be implemented as a generic training loop that writes the epoch number, training loss and validation accuracy to console at the end of each epoch. 
6. After training, measure the classification accuracy of the best performing model on the test dataset. Write the best validation accuracy and the test accuracy to console as well.

Running your `linear_cats_and_dogs.py` script should produce (ignoring the values) a console output similar to:
```python
...
epoch 99
train loss: 0.636
val acc: 0.595
epoch 100
train loss: 0.635
val acc: 0.596
--------------------
val acc (best): 0.596
test acc: 0.616
```

## Report

Write a short report (2 to 3 pages) that answers the following questions:

* Why do general machine learning algorithms (those expecting vector input) perform
poorly on images? What is a feature, and what is the purpose of feature extraction?
Explain the terms low-level feature and high-level feature.
* What is the purpose of a loss function? What does the cross-entropy measure? Which
criteria must the ground-truth labels and predicted class-scores fulfill to support the
cross-entropy loss, and how is this ensured?
* What is the purpose of the training, validation, and test sets and why do we need all of them?

Also include your results obtained from `linear_cats_and_dogs.py`. Include the validation accuracies as a table or (better) a plot as well as the final test accuracy. Compare the best validation accuracy and the final test accuracy, and discuss the results. Furthermore, state which optimizer (and optimizer parameters) were used.

## Submission

Submit your assignment until **April 15th at 11pm**. To do so, create a zip archive including the report, the complete `dlvc` folder with your implementations as well as `linear_cats_and_dogs.py` (do not include the CIFAR-10 dataset). More precisely, after extracting the archive we should obtain the following:

    group_x/
        report.pdf
        linear_cats_and_dogs.py
        dlvc/
            batches.py
            ...
            datasets/
                ...
            ...

Submit the zip archive in TUWEL. Make sure you've read the general assignment information [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/blob/main/assignments/general.md) before your final submission.
