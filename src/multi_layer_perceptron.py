import joblib
from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptron:

    def __init__(self, name, hidden_layers_size, solver, activation_function, learning_rate_init,
                 learning_rate, momentum, optimisation_tolerance, num_iterations_no_change, max_iterations, verbose):
        self.name = name
        self.input_data = list()
        self.target_data = None
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

    def train(self):
        self.mlp.fit(self.input_data, self.target_data)

    def make_predictions(self):
        for x in self.input_data.values:
            print("{} prediction = {} ({})".format(x, self.mlp.predict([x]), self.mlp.predict_proba([x])))

    def save_trained_nn(self):
        joblib.dump(self.mlp, "../neural_networks/{}.joblib".format(self.name))
