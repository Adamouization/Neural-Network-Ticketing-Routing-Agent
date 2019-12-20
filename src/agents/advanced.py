import time

from agents.common import encode_data, run_multi_layer_perceptron
import config as config
from helpers import print_runtime
from neural_network.data_processor import DataProcessor
from neural_network.multi_layer_perceptron import MultiLayerPerceptron


def run_advanced_agent():
    """
    Runs the basic agent program.
    :return:
    """
    # Retrieve data from CSV file and encode it.
    data = DataProcessor(config.csv_file)
    encode_data(data)

    mlp = MultiLayerPerceptron(
        classifier_model="dt",
        name=config.csv_file,
        input_data=data.input_data_encoded,
        target_data=data.target_data_encoded,
    )
    start_time = time.time()  # Start measuring runtime.
    run_multi_layer_perceptron(mlp)
    print_runtime("Training", round(time.time() - start_time, 2))  # Record and print runtime.
