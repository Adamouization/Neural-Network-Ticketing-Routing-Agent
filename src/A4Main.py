from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier


def main():
    solver = 'sgd'  # Stochastic Gradient Descent.
    learning_rate = 'constant'  # Default.
    learning_rate_init = 0.5  # Learning rate.
    hidden_layer_sizes = (2,)  # Length = n_layers - 2. We only need 1 (n_units,).
    activation = 'logistic'  # Logistic sigmoid activation function.
    momentum = 0.3  # Momentum.
    verbose = True  # To see the iterations.

    # When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations,
    # convergence is considered to be reached and training stops.
    tol = 0.0001
    n_iter_no_change = 5000  # Number of iterations with no change.
    max_iter = 10000  # Maximum number of iterations.

    # hidden units 2
    clf = MLPClassifier(
        solver=solver,
        learning_rate_init=learning_rate_init,
        hidden_layer_sizes=hidden_layer_sizes,
        verbose=verbose,
        momentum=momentum,
        activation=activation,
        n_iter_no_change=n_iter_no_change,
        max_iter=max_iter
    )

    x = [[1, 1], [0, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 1]

    # Training & Checking results
    clf.fit(x, y)

    h = clf.predict([[0, 0]])
    k = clf.predict_proba([[0, 0]])
    print(h)
    print(k)

    joblib.dump(clf, 'mynetwork.joblib')


if __name__ == "__main__":
    main()
