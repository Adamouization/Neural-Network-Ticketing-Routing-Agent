import joblib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import src.config as config
from src.data import inverse_encoding, inverse_encoding_no_categories

# Global variables
fontsize = 12


class MultiLayerPerceptron:

    def __init__(self, name, input_data, target_data, hidden_layers_size, solver, activation_function,
                 learning_rate_init, learning_rate, momentum, optimisation_tolerance, num_iterations_no_change,
                 max_iterations, verbose):
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
            learning_rate=learning_rate,
            momentum=momentum,
            tol=optimisation_tolerance,
            n_iter_no_change=num_iterations_no_change,
            max_iter=max_iterations,
            verbose=verbose,
        )

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.input_data,
                                                                                self.target_data,
                                                                                test_size=0.20)
        self.categories = list(self.y_test.columns)

    def train(self):
        self.mlp.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.predictions = self.mlp.predict(self.X_test)
        probability_predictions = self.mlp.predict_proba(self.X_test)
        if config.debug:
            for i, p in enumerate(inverse_encoding(self.predictions)):
                print("{}\nPrediction = {}: {}\n".format(self.X_test.iloc[i].to_frame().T,
                                                         self.categories[p],
                                                         probability_predictions[i].tolist()))

    def show_results(self):
        # Plot error loss curve.
        plt.plot(self.mlp.loss_curve_)
        plt.xlabel("Epochs", fontsize=fontsize)
        plt.ylabel("Error loss", fontsize=fontsize)
        plt.show()

        # Calculate confusion matrix
        ground_truth_values = inverse_encoding(self.y_test)
        estimated_target_values = inverse_encoding_no_categories(self.predictions, self.categories)
        cm = confusion_matrix(ground_truth_values, estimated_target_values)
        # Convert numpy array to pandas DataFrame
        cm = pd.DataFrame(data=cm, index=[i for i in self.categories], columns=[i for i in self.categories])

        # Display confusion matrix as a heat map.
        sn.heatmap(cm, cmap="YlGnBu", annot=True, annot_kws={"size": fontsize})
        plt.xlabel("Predictions", fontsize=fontsize)
        plt.ylabel("Ground truth values", fontsize=fontsize)
        plt.show()

        if config.debug:
            # Generate classification report and print confusion matrix and report to command line.
            cr = classification_report(ground_truth_values, estimated_target_values)
            print(cm)
            print()
            print(cr)

    def save_trained_nn(self):
        joblib.dump(self.mlp, "../neural_networks/{}.joblib".format(self.name))
