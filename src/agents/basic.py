import time

from agents.common import encode_data, run_multi_layer_perceptron
import config as config
from helpers import print_runtime
from neural_network.data_processor import DataProcessor
from neural_network.grid_search_algorithm import GridSearch
from neural_network.multi_layer_perceptron import MultiLayerPerceptron


def run_basic_agent():
    """
    Runs the basic agent program.
    :return:
    """
    # Retrieve data from CSV file and encode it.
    data = None
    if config.csv_file == "tickets":
        data = DataProcessor(config.csv_file)
    elif config.csv_file == "AND_gate":
        data = DataProcessor("AND_gate")
    elif config.csv_file == "OR_gate":
        data = DataProcessor("OR_gate")
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
        start_time = time.time()  # Start measuring runtime.
        run_multi_layer_perceptron(mlp)
        print_runtime("Training", round(time.time() - start_time, 2))  # Record and print runtime.

        # Uncomment below to calculate an aggregate confusion matrix over 5 runs.
        # aggregate_cm(data, runs=5)


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
    )


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


def aggregate_cm(data, runs):
    """
    Standalone script to calculate an aggregate confusion matrix.
    :param runs: Number of iterations.
    :param data: DataProcessor instance.
    :return: None
    """
    import seaborn as sn
    import matplotlib.pyplot as plt
    aggregated_cm = None
    title = ""
    for count in range(0, runs, 1):
        mlp = create_multi_layer_perceptron(data)
        mlp.split_data()
        mlp.train()
        mlp.test()
        mlp.show_results()
        mlp.save_trained_nn()
        cm = mlp.get_confusion_matrix()
        if count == 0:  #
            aggregated_cm = cm
            aggregated_cm[:] = 0
            title = "hls{}-{}-{}-a{}-m{}.csv".format(mlp.mlp.hidden_layer_sizes, mlp.mlp.solver, mlp.mlp.activation,
                                                     mlp.mlp.learning_rate_init, mlp.mlp.momentum)
        # Calculate aggregate.
        for i, row in enumerate(cm.values):
            for j, val in enumerate(row):
                val = aggregated_cm.iloc[i][j] + val
                aggregated_cm.iloc[i][j] = val
    # Plot aggregate confusion matrix in heatmap.
    fig, ax = plt.subplots()
    sn.heatmap(aggregated_cm, cmap="YlGnBu", annot=True, annot_kws={"size": 12})
    plt.xlabel("Predictions", fontsize=12)
    plt.ylabel("Ground truth values", fontsize=12)
    ax.set_title(title, fontsize=8)
    plt.show()
