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

Download the reference code from [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/reference), making sure that the file structure is preserved, and rename the root folder to something other than `reference`. Read the code and make sure that you understand what the individual classes and methods are doing.

[Download](https://www.cs.toronto.edu/~kriz/cifar.html) and extract the *Python* version of the CIFAR10 dataset somewhere *outside* the code folder. Read the website to understand which classes there are and how the data are structured and read.

Then implement the `PetsDataset` (`datasets/pets.py`). Some parts of this assignment will be tested automatically so make sure to follow the instructions in the code exactly (note, for example the specified data types). Make sure the following applies. If not, you made a mistake somewhere:

* Number of samples in the individual datasets: 7959 (training), 2041 (validation), 2000 (test).
* Total number of cat and dog samples: 6000 per class
* Image shape: always `(32, 32, 3)`, image type: always `uint8`
* Labels of first 10 training samples: `0 0 0 0 1 0 0 0 0 1`
* Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (`cv2.imshow`, `cv2.imwrite`).

Do not modify, add or delete any other files.
