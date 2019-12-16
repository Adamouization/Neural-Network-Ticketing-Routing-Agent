import joblib

from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import src.config as config
from .data_processor import inverse_encoding, inverse_encoding_no_categories


class MultiLayerPerceptron:
    """
    Class containing data necessary to train and test a neural network, as well as functions to split the data, train
    the neural network, test it by making predictions and plotting evaluation results.
    """

    def __init__(self, name, input_data, target_data, hidden_layers_size=(15,), solver="adam",
                 activation_function="logistic", learning_rate_init=0.6, momentum=0.9, optimisation_tolerance=0.0001,
                 num_iterations_no_change=1000, max_iterations=10000, verbose=config.debug):
        """
        Initialise variables and create a new neural network using SciKit's MLPClassifier class. Default neural network
        hyperparameters are from the optimal solution generate from the Grid Search algorithm. By default, the network
        has a single hidden layer with 15 hidden units, uses a Stochastic Gradient Descent Optimiser as a solver and a
        logistic sigmoid activation function. The learning rate is at 0.6 and the momentum at 0.9. Stopping conditions
        include an optimisation tolerance of 0.0001 after 1000 iterations with no changes and a maximum number of
        iterations of 10000.

        :param name: The name of the neural network (based on the CSV file).
        :param input_data: The input data.
        :param target_data: The target data.
        :param hidden_layers_size: The number of hidden layers and number of units per layer.
        :param solver: The method used to train the neural network e.g. Stochastic Gradient Descent.
        :param activation_function: The activation function used e.g. sigmoid, ReLU or tanh.
        :param learning_rate_init: The initial learning rate constant to speedup training (remains constant).
        :param momentum: The momentum constant used to accelerate training and overcome problems like local minima.
        :param optimisation_tolerance:  The tolerance, which stops the training when it does not improving by this
                                        amount for num_iterations_no_change.
        :param num_iterations_no_change: The number of iterations with no change above the optimisation_tolerance.
        :param max_iterations: The maximum number of iterations before ending training.
        :param verbose: Used to debug. Prints error at each iteration when set to True.
        """
        self.name = name
        self.input_data = input_data
        self.target_data = target_data
        self.X_train, self.X_test, self.y_train, self.y_test = (int(),) * 4
        self.categories = list()
        self.predictions = None
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers_size,
            solver=solver,
            activation=activation_function,
            learning_rate_init=learning_rate_init,
            momentum=momentum,
            tol=optimisation_tolerance,
            n_iter_no_change=num_iterations_no_change,
            max_iter=max_iterations,
            verbose=verbose,
        )

    def split_data(self):
        """
        Split the input and target data for training and testing with an 80%/20% split. Determines the list categories
        used when processing the results.
        :return: None
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.input_data,
                                                                                self.target_data,
                                                                                test_size=0.20,
                                                                                stratify=self.target_data)
        self.categories = list(self.y_test.columns)

    def train(self):
        """
        Trains the neural network using the training data.
        :return:
        """
        self.mlp.fit(self.X_train, self.y_train)

    def test(self):
        """
        Test by making predictions based on the testing data input.
        :return:
        """
        self.predictions = self.mlp.predict(self.X_test)
        probability_predictions = self.mlp.predict_proba(self.X_test)
        if config.debug:
            for i, p in enumerate(inverse_encoding(self.predictions)):
                print("{}\nPrediction = {}: {}\n".format(self.X_test.iloc[i].to_frame().T,
                                                         self.categories[p],
                                                         probability_predictions[i].tolist()))

    def show_results(self):
        """
        Plots the training and testing results, starting with a plot of the error loss function w.r.t. the number of
        epochs, followed by the confusion matrix of the testing predictions plotted in a heatmap.
        :return: None
        """
        fontsize = 12

        # Plot error loss curve.
        fig, ax = plt.subplots()
        ax.plot(self.mlp.loss_curve_)
        plt.xlabel("Epochs", fontsize=fontsize)
        plt.ylabel("Error loss", fontsize=fontsize)
        anchored_text = AnchoredText(
            "Epochs: {}\nError: {}".format(len(self.mlp.loss_curve_), round(self.mlp.loss_curve_[-1], 5)),
            loc='upper right', prop=dict(size=8), frameon=True)
        anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.set_title(
            "hidden_layer_sizes:{} solver:{} activation:{} learning_rate_init:{} momentum:{}".format(
                self.mlp.hidden_layer_sizes, self.mlp.solver, self.mlp.activation, self.mlp.learning_rate_init,
                self.mlp.momentum
            ),
            fontsize=8
        )
        ax.add_artist(anchored_text)
        plt.show()

        # Calculate confusion matrix
        ground_truth_values = inverse_encoding(self.y_test)
        estimated_target_values = inverse_encoding_no_categories(self.predictions, self.categories)
        cm = confusion_matrix(ground_truth_values, estimated_target_values)

        accuracy = _calculate_accuracy(cm)
        print("Accuracy: {}%".format(accuracy))

        # Convert confusion matrix from numpy array to pandas DataFrame
        cm = pd.DataFrame(data=cm, index=[i for i in self.categories], columns=[i for i in self.categories])

        # Display confusion matrix as a heat map.
        fig, ax = plt.subplots()
        sn.heatmap(cm, cmap="YlGnBu", annot=True, annot_kws={"size": fontsize})
        plt.xlabel("Predictions", fontsize=fontsize)
        plt.ylabel("Ground truth values", fontsize=fontsize)
        ax.set_title("Accuracy: {}%".format(accuracy), fontsize=fontsize)
        plt.show()

        if config.debug:
            # Generate classification report and print confusion matrix and report to command line.
            cr = classification_report(ground_truth_values, estimated_target_values)
            print(cm)
            print()
            print(cr)

    def save_trained_nn(self):
        """
        Saves the trained neural network and its weights in a file for future re-use.
        :return: None
        """
        joblib.dump(self.mlp, "../neural_networks/{}.joblib".format(self.name))


def _calculate_accuracy(cm):
    """
    Calculates the accuracy of the testing using the results from the confusion matrix (counting the number of true
    positives)
    :param cm: The confusion matrix.
    :return: The accuracy in percentage form.
    """
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return (diagonal_sum / sum_of_all_elements) * 100
