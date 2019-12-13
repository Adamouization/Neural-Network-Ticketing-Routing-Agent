import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptron:

    def __init__(self, name, hidden_layers_size, solver, activation_function, learning_rate_init,
                 learning_rate, momentum, optimisation_tolerance, num_iterations_no_change, max_iterations, verbose):
        self.name = name
        self.input_data = list()
        self.target_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = (int(),) * 4
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

    def train(self):
        self.mlp.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.predictions = self.mlp.predict(self.X_test)
        print(self.predictions)
        print(self.mlp.predict_proba(self.X_test))
        # for x in self.input_data.values:
        #     print("{} prediction = {} ({})".format(x, self.mlp.predict([x]), self.mlp.predict_proba([x])))

    def show_results(self):
        plt.plot(self.mlp.loss_curve_)
        plt.xlabel("Epochs")
        plt.ylabel("Error loss")
        plt.show()
        # print(confusion_matrix(self.y_test, self.predictions))
        # print(classification_report(self.y_test, self.predictions))

    def save_trained_nn(self):
        joblib.dump(self.mlp, "../neural_networks/{}.joblib".format(self.name))
