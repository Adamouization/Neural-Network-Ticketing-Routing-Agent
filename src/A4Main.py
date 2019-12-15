import argparse

import src.config as config
from src.data_processor import Data
from src.grid_search_algorithm import GridSearch
from src.multi_layer_perceptron import MultiLayerPerceptron


def main():
    """
    Program entry point. Parses command line arguments to decide which agent to run and what data to use.
    :return: None
    """
    parse_command_line_arguments()

    # Run the Basic agent to train the neural network given the data or to determine its optimal parameters.
    if config.agent == "Bas":

        # Retrieve data from CSV file and encode it.
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

        # Run Grid Search algorithm to determine optimal hyperparameters for the neural network.
        if config.is_grid_search:
            gs = GridSearch(data.input_data_encoded, data.target_data_encoded)
            run_grid_search(gs)
        # Train and test the neural network based on the chosen data.
        else:
            mlp = create_multi_layer_perceptron(data)
            run_multi_layer_perceptron(mlp)

    # Run the Intermediate agent to interact with the user through a text-based interface and make early predictions.
    elif config.agent == "Int":
        print("Intermediate Agent")

    # Run the Advanced agent.
    elif config.agent == "Adv":
        print("Advanced Agent")

    # Error in agent specified.
    else:
        print("Wrong agent specified. Please use --agent 'Bas', 'Int', or 'Adv'")
        exit(1)


def parse_command_line_arguments():
    """
    Parse command line arguments and save them in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent",
                        required=True,
                        help="The type of agent to run (Basic 'Bas', Intermediate 'Int' or Advanced 'Adv')."
                        )
    parser.add_argument("-c", "--csv",
                        required=True,
                        help="The CSV data used to train and test the neural network. Choose from the data available "
                             "in the data directory"
                        )
    parser.add_argument("-g", "--gridsearch",
                        action="store_true",
                        help="Include this flag to run the grid search algorithm to determine the optimal "
                             "hyperparameters for the neural network.")
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Include this flag additional print statements and data for debugging purposes.")
    args = parser.parse_args()
    config.agent = args.agent
    config.csv_file = args.csv
    config.is_grid_search = args.gridsearch
    config.debug = args.debug


def encode_data(data):
    """
    Encode the input and target data to be used by the neural network.
    :param data: The raw CSV data.
    :return: None
    """
    data.encode_input_data()
    data.encode_target_data()


def create_multi_layer_perceptron(data):
    """
    Creates a new instance of the MultiLayerPerceptron class with optimal hyperparameters (which were determined in
    previous runs by analysing the results of the grid search algorithm.
    :param data: The data object containing the encoded input and target data.
    :return: The instantiated neural network.
    """
    return MultiLayerPerceptron(
        name=config.csv_file,
        input_data=data.input_data_encoded,
        target_data=data.target_data_encoded,
        hidden_layers_size=(9,),        # Single hidden layer with 9 hidden units.
        solver='adam',                  # Stochastic Gradient Descent Optimiser.
        activation_function='tanh',     # Hyperbolic tan activation function.
        learning_rate_init=0.3,         # Learning rate.
        momentum=0.7,                   # Momentum.
        optimisation_tolerance=0.0001,  # Stop condition: score not improving by tol for num_iterations_no_change.
        num_iterations_no_change=100,   # Stop condition: number of iterations with no change.
        max_iterations=10000,           # Stop condition: maximum number of iterations.
        verbose=config.debug,           # Print iterations at each step.
    )


def run_multi_layer_perceptron(mlp):
    """
    Execution flow to train and test the neural network.
    :param mlp: The neural network object.
    :return: None
    """
    mlp.split_data()
    mlp.train()
    mlp.test()
    mlp.show_results()
    mlp.save_trained_nn()


def run_grid_search(gs):
    """
    Execution flow to run the grid search algorithm and determine the optimal hyperparameters.
    :param gs: The grid search object.
    :return: None
    """
    gs.split_data()
    gs.run_grid_search()
    gs.print_optimal_parameters()
    gs.save_grid_search()


if __name__ == "__main__":
    main()
