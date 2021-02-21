## Digit Recognition with Python

The app uses a trained neural network written from scratch to guess handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
It has a tkinter GUI to interact with the network.
It scored a 96.98% accuracy on the test examples provided by MNIST.
It was written after watching Andrew Ng's course on Coursera
#### Things used
* NumPy
* PIL
* Math :)

#### How to use
1. Run the file digit_recogniser.py
2. Install any missing modules with pip install "module name"
3. Try to draw inside the tkinter window

#### If you want to train the network yourself
1. Download the processed data from https://mega.nz/folder/GmIxQazJ#Y6DTKIRNmAx1eEnLrCUrVA
2. Place the data in a data folder in the same folder as the file NeuralNetwork.py
3. In the NeuralNetwork.py file run NeuralNetwork.train() which will every 100 iterations save the trained parameters
