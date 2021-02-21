import numpy as np
from progress_bar import progress_bar
import math
from matplotlib import pyplot as plt


class NeuralNetwork:

    def __init__(self, theta=np.array([])):
        np.random.seed(10)
        self.images = np.load("./data/images.npy")
        self.labels = np.load("./data/labels.npy")
        self.images_test = np.load("./data/images_test.npy")
        self.labels_test = np.load("./data/labels_t.npy")
        # Number of training examples
        self.m = 60000
        # Number of neurons in the first and second hidden layer
        self.neurons1 = 50
        self.neurons2 = 50
        # Regularization parameter
        self.regul_lambda = 0.3
        # Learning rate
        self.alpha = 1

        self.THETA = theta
        # Epsilon for small values of weights
        self.epsilon_init1 = (math.sqrt(6)) / (math.sqrt(785 + 50))
        self.epsilon_init2 = (math.sqrt(6)) / (math.sqrt(51 + 50))
        self.epsilon_init3 = (math.sqrt(6)) / (math.sqrt(51 + 10))
        # Matrix of weights and biases for the first layer
        self.THETA1 = np.random.rand(785, self.neurons1) * 2 * self.epsilon_init1 - self.epsilon_init1
        self.a1 = self.sigmoid(np.matmul(self.images, self.THETA1))
        # First hidden layer
        self.hidden_layer1 = np.insert(self.a1, 0, 1, axis=1)
        # Matrix of weight and biases for the second layer
        self.THETA2 = np.random.rand(self.neurons1 + 1, self.neurons2) * 2 * self.epsilon_init2 - self.epsilon_init2
        self.a2 = self.sigmoid(np.matmul(self.hidden_layer1, self.THETA2))
        # Second hidden layer
        self.hidden_layer2 = np.insert(self.a2, 0, 1, axis=1)
        # Matrix of weights and biases for the third layer
        self.THETA3 = np.random.rand(self.neurons2 + 1, 10) * 2 * self.epsilon_init3 - self.epsilon_init3
        # Predictions of the NN
        self.output_layer = self.sigmoid(np.matmul(self.hidden_layer2, self.THETA3))
        self.THETA = np.array([self.THETA1, self.THETA2, self.THETA3], dtype=object)

        # For plotting
        self.costs = np.array([])

    def sigmoid(self, arr):
        return 1 / (1 + np.exp(-arr))

    def calculate_output(self, THETA, arr):
        T1 = THETA[0]
        T2 = THETA[1]
        T3 = THETA[2]

        z1 = self.sigmoid(np.matmul(arr, T1))
        a1 = np.insert(z1, 0, 1, axis=1)

        z2 = self.sigmoid(np.matmul(a1, T2))
        a2 = np.insert(z2, 0, 1, axis=1)

        a3 = self.sigmoid(np.matmul(a2, T3))

        return a3

    def cost(self, THETA):
        # J in math notation
        cost = 0
        h = self.calculate_output(THETA, self.images)
        for i in range(len(self.output_layer)):
            # Network's prediction for a given example
            h_example = h[i]
            # True label for a given example
            y = self.labels[i]
            example_cost = np.dot(-y.T, np.log(h_example)) - np.dot((1 - y.T), np.log(1 - h_example))
            cost += example_cost

        regularization = 0
        # Summing over all the matrix elements squared
        for i in THETA[0][1:]:
            for j in i:
                regularization += j ** 2

        for i in THETA[1][1:]:
            for j in i:
                regularization += j ** 2

        for i in THETA[2][1:]:
            for j in i:
                regularization += j ** 2

        cost = cost / self.m + (self.regul_lambda / (2 * self.m)) * regularization
        return cost

    def backpropagation(self):
        grad_dim1 = np.zeros((785, 50))
        grad_dim2 = np.zeros((51, 50))
        grad_dim3 = np.zeros((51, 10))
        for i in range(self.m):
            ex = self.images[i]
            # First layer forward prop
            z2 = np.matmul(self.THETA1.T, ex)
            a2 = self.sigmoid(z2)
            a2 = np.insert(a2, 0, 1, axis=0)
            # Second layer forward prop
            z3 = np.matmul(self.THETA2.T, a2)
            a3 = self.sigmoid(z3)
            a3 = np.insert(a3, 0, 1, axis=0)
            # Output layer forward prop
            z4 = np.matmul(self.THETA3.T, a3)
            a4 = self.sigmoid(z4)
            # Backprop first layer
            delta4 = a4 - self.labels[i]
            # Backprop second layer
            delta3 = np.multiply(np.matmul(self.THETA3, delta4), np.multiply(a3, 1 - a3))
            # Backprop third layer
            delta2 = np.multiply(np.matmul(self.THETA2, delta3[1:]), np.multiply(a2, 1 - a2))
            grad1 = np.matmul(ex.reshape(785, 1), delta2[1:].reshape(1, 50))
            grad2 = np.matmul(a2.reshape(51, 1), delta3[1:].reshape(1, 50))
            grad3 = np.matmul(a3.reshape(51, 1), delta4.reshape(1, 10))
            grad_dim1 += grad1
            grad_dim2 += grad2
            grad_dim3 += grad3
            progress_bar(100, int(i / (self.m / 100)), "Backpropagation")
        print("")
        grad_dim1 = (1 / self.m) * grad_dim1
        grad_dim2 = (1 / self.m) * grad_dim2
        grad_dim3 = (1 / self.m) * grad_dim3
        grad_dim1[1:] = grad_dim1[1:] + (self.regul_lambda / self.m) * self.THETA1[1:]
        grad_dim2[1:] = grad_dim2[1:] + (self.regul_lambda / self.m) * self.THETA2[1:]
        grad_dim3[1:] = grad_dim3[1:] + (self.regul_lambda / self.m) * self.THETA3[1:]
        return np.array([grad_dim1, grad_dim2, grad_dim3], dtype=object)

    def train(self):
        for i in range(10000):
            print("Run: ", i + 1)
            grads = self.backpropagation()
            past_cost = self.cost(self.THETA)
            np.append(self.costs, past_cost)
            self.THETA1 = self.THETA1 - self.alpha * grads[0]
            self.THETA2 = self.THETA2 - self.alpha * grads[1]
            self.THETA3 = self.THETA3 - self.alpha * grads[2]
            self.THETA = np.array([self.THETA1, self.THETA2, self.THETA3], dtype=object)

            current_cost = self.cost(self.THETA)
            cost_difference = past_cost - current_cost
            print("Current cost: ", current_cost)
            print("Cost difference: ", cost_difference)
            print("-" * 80)

            if i % 100 == 0:
                np.save("Theta{}.npy".format(int(i / 100)), self.THETA)

            if cost_difference < 0.000001:
                np.save("Theta.npy", self.THETA)
                break

        np.save("Theta.npy", self.THETA)
        plt.title("Cost / number of iterations")
        x = np.arange(1, 10000, 1)
        plt.plot(x, self.costs)
        plt.show()

    def test_network(self):
        num_of_errors = 0
        hypothesis = self.calculate_output(self.THETA, self.images_test)
        for i in range(10000):
            # Turns output into [0, 0, 0, 1] notation where the 1 is the prediction of the network
            example = hypothesis[i]
            example_output = np.zeros((hypothesis[i].shape), dtype=int)
            example_output[np.argmax(example)] = 1
            if not np.array_equal(example_output, self.labels_test[i]):
                num_of_errors += 1
        print("Accuracy report...")
        print("Num of errors: ", num_of_errors, "/10000", sep="")
        print("Accuracy: ", ((10000 - num_of_errors) / 10000) * 100, "%")
