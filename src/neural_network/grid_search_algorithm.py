import time

import pandas as pd
from pyspin.spin import Box1, make_spin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

# Grid Search parameters to test (single parameters used to overwrite default MLPClassifier values, not tested).
grid_search_params = {
    'hidden_layer_sizes': [(3,), (5,), (7,), (9,), (15,), (25,)],
    'solver': ["sgd", "adam"],
    'activation': ["logistic", "relu", "tanh"],
    'learning_rate_init': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
    'tol': [1, 0.1, 0.01, 0.001, 0.0001],
    'n_iter_no_change': [100],  # Param not being tested.
    'max_iter': [10000]         # Param not being tested.
}


def _number_unique_parameters_combination():
    """
    Calculates the number of parameter combinations to test to have an idea how long the search will take.
    :return: the total number of combinations to test
    """
    combinations = 1
    for key in grid_search_params:
        combinations = combinations * len(grid_search_params[key])
    return combinations


class GridSearch:
    """
    Class implementing the Grid Search algorithm, the data necessary to train and test the neural network with all
    parameter combinations, as well as functions to split the data, print the optimal result and save the raw results in
    a CSV file.
    """

    def __init__(self, input_data, target_data):
        """
        Initialise variables and SciKit's GridSearch algorithm.
        :param input_data: The input data.
        :param target_data: The target data.
        """
        self.input_data = input_data
        self.target_data = target_data
        self.grid_search = GridSearchCV(estimator=MLPClassifier(),
                                        param_grid=grid_search_params,
                                        scoring='accuracy',
                                        n_jobs=-1)
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def split_data(self):
        """
        Split the input and target data for training and testing with an 80%/20% split.
        :return: None.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.input_data,
                                                                                self.target_data,
                                                                                test_size=0.20,
                                                                                stratify=self.target_data)

    @make_spin(
        Box1, "Running grid search algorithm on Multi-Layer Perceptron with {} unique combinations...".format(
            _number_unique_parameters_combination()))
    def run_grid_search(self):
        """
        Runs the grid search algorithm, testing every possible combination in the grid_search_params dictionnary.
        Measures runtime as well.
        :return: None.
        """
        # Start measuring runtime.
        start_time = time.time()

        # Run Grid Search using all possible combinations in grid_search_params.
        self.grid_search.fit(self.X_train, self.y_train)

        # Record and print runtime.
        runtime = round(time.time() - start_time, 2)
        print("\n--- Grid Search Runtime: {} seconds ---".format(runtime))

    def print_optimal_parameters(self):
        """
        Prints the optimal hyperparameters found and the testing score achieved with those parameters.
        :return: None.
        """
        print("Optimal hyperparameters found: ".format(self.grid_search.best_params_))
        print("\nAccuracy with hyperparameters above: ".format(self.grid_search.best_score_))

    def save_grid_search(self):
        """
        Dumps the raw results for each combination into a CSV file for further analysis in tools such as Excel.
        :return: None.
        """
        df = pd.DataFrame(self.grid_search.cv_results_)
        df.to_csv("../results/grid_search_results.csv")
