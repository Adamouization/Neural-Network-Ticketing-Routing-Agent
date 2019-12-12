import pandas as pd

import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


def main():
    solver = 'sgd'  # Stochastic Gradient Descent.
    learning_rate = 'constant'  # Doesn't increase the learning rate
    learning_rate_init = 0.5  # Learning rate.
    hidden_layer_sizes = (2,)  # Length = n_layers - 2. We only need 1 (n_units,).
    activation_function = 'logistic'  # Sigmoid activation function.
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
        activation=activation_function,
        n_iter_no_change=n_iter_no_change,
        max_iter=max_iter,
        learning_rate=learning_rate
    )

    # Data
    names = ["Request", "Incident", "WebServices", "Login", "Wireless", "Printing", "IdCards", "Staff", "Students",
             "Response Team"]
    tickets_data = pd.read_csv("../data/tickets.csv", names=names, skiprows=[0])

    # X = [[1, 1], [0, 0], [1, 0], [0, 1]]
    X = tickets_data.iloc[:, 0:tickets_data.columns.size - 1]  # Data from first 9 columns.
    print(X)

    # Target output.
    # y = [0, 0, 1, 1]
    y = tickets_data.iloc[:, -1]  # Data from last column
    print(y)

    # Integer encode
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    # Binary encode
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # Invert first example
    inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    print(inverted)

    # Training & Checking results
    # clf.fit(x, y)

    # h = clf.predict([[0, 0]])
    # k = clf.predict_proba([[0, 0]])
    # print(h)
    # print(k)

    # joblib.dump(clf, 'mynetwork.joblib')


if __name__ == "__main__":
    main()
