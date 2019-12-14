import argparse

import src.config as config
from src.data import Data
from src.multi_layer_perceptron import MultiLayerPerceptron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv",
                        required=True,
                        help="The CSV data used to train and test the neural network. Choose from the data available "
                             "in the data directory"
                        )
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Include this flag additional print statements and data for debugging purposes.")
    args = parser.parse_args()
    config.csv_file = args.csv
    config.debug = args.debug

    data = None
    if config.csv_file == "tickets":
        data = Data(config.csv_file)
    elif config.csv_file == "AND_gate":
        data = Data("AND_gate")
    elif config.csv_file == "OR_gate":
        data = Data("OR_gate")
    else:
        print("CSV file {} could not be found, please check spelling.")
        exit(1)

    encode_data(data)
    mlp = create_multi_layer_perceptron(data)
    use_multi_layer_perceptron(mlp)


def encode_data(data):
    data.encode_input_data()
    data.encode_target_data()


def create_multi_layer_perceptron(data):
    return MultiLayerPerceptron(
        name=config.csv_file,
        input_data=data.input_data_encoded,
        target_data=data.target_data_encoded,
        hidden_layers_size=(7,),         # Length = n_layers - 2. We only need 1 (n_units,)
        solver='sgd',                    # Stochastic Gradient Descent
        activation_function='logistic',  # Sigmoid activation function
        learning_rate='constant',        # Don't increase learning rate during training
        learning_rate_init=0.7,          # Learning rate
        momentum=0.5,                    # Momentum
        optimisation_tolerance=0.0001,   # Stop condition: score not improving by this much for num_iterations_no_change
        num_iterations_no_change=500,    # Stop condition: number of iterations with no change
        max_iterations=10000,            # Stop condition: maximum number of iterations.
        verbose=config.debug,            # Print iterations at each step
    )


def use_multi_layer_perceptron(mlp):
    mlp.split_data()
    mlp.train()
    mlp.make_predictions()
    mlp.show_results()
    mlp.save_trained_nn()


if __name__ == "__main__":
    main()
