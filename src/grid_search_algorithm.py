import time

import pandas as pd
from pyspin.spin import Box1, make_spin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

grid_search_params = {
    'hidden_layer_sizes': [(3,), (5,), (7,), (9,), (20,)],
    'solver': ["sgd", "adam"],
    'activation': ["logistic", "relu", "tanh"],
    'learning_rate_init': [0.1, 0.3, 0.7, 0.9],
    'momentum': [0.1, 0.4, 0.7],
    'tol': [0.1, 0.01, 0.001, 0.0001],
    'n_iter_no_change': [100],
    'max_iter': [5000]
}


def _number_unique_parameters_combination():
    combinations = 1
    for key in grid_search_params:
        combinations = combinations * len(grid_search_params[key])
    return combinations


class GridSearch:

    def __init__(self, input_data, target_data):
        _number_unique_parameters_combination()
        self.grid_search = GridSearchCV(estimator=MLPClassifier(),
                                        param_grid=grid_search_params,
                                        scoring='accuracy',
                                        n_jobs=-1)
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

        self.split_data(input_data, target_data)
        self.run_grid_search()
        self.print_optimal_parameters()
        self.save_grid_search()

    def split_data(self, input_data, target_data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(input_data,
                                                                                target_data,
                                                                                test_size=0.20)

    @make_spin(
        Box1, "Running grid search algorithm on Multi-Layer Perceptron with {} unique combinations...".format(
            _number_unique_parameters_combination()))
    def run_grid_search(self):
        # Start measuring runtime.
        start_time = time.time()

        # Run Grid Search using all possible combinations in grid_search_params.
        self.grid_search.fit(self.X_train, self.y_train)

        # Record and print runtime.
        runtime = round(time.time() - start_time, 2)
        print("\n--- Grid Search Runtime: {} seconds ---".format(runtime))

    def print_optimal_parameters(self):
        print(self.grid_search.best_params_)
        print(self.grid_search.best_score_)

    def save_grid_search(self):
        df = pd.DataFrame(self.grid_search.cv_results_)
        df.to_csv("../results/grid_search_results.csv")
