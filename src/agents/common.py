initial_data_points = 250


def encode_data(data):
    """
    Encode the input and target data to be used by the neural network.
    :param data: The raw CSV data.
    :return: None
    """
    data.encode_input_data()
    data.encode_target_data()


def run_multi_layer_perceptron(mlp, is_refitted_nn=False):
    """
    Execution flow to train and test the neural network. If the neural network has been retrained, moves data around to
    ensure that new data is included in the training set and not the testing set.
    :param is_refitted_nn: Boolean specifying if the neural network has already been retrained, which may lead to
    unbalanced data.
    :param mlp: The neural network object.
    :return: None
    """
    mlp.split_data()

    # Check if new rows are in training, else move them around.
    if is_refitted_nn:
        data_training_indexes = mlp.X_train.index.values  # Rows that are in the training set.
        if initial_data_points not in data_training_indexes:
            print("New data split into training set.")  # Todo: move data over

    mlp.train()
    mlp.test()
    mlp.show_results()
    mlp.save_trained_nn()
