from src.data import Data
from src.multi_layer_perceptron import MultiLayerPerceptron


def main():

    mlp = MultiLayerPerceptron(
        name="AND gate",
        hidden_layers_size=(2,),  # Length = n_layers - 2. We only need 1 (n_units,)
        solver='sgd',  # Stochastic Gradient Descent
        activation_function='logistic',  # Sigmoid activation function
        learning_rate='constant',  # Don't increase learning rate during training
        learning_rate_init=0.5,  # Learning rate
        momentum=0.3,  # Momentum
        optimisation_tolerance=0.001,  # Stop condition: score not improving by this much for num_iterations_no_change
        num_iterations_no_change=5000,  # Stop condition: number of iterations with no change
        max_iterations=10000,  # Stop condition: maximum number of iterations.
        verbose=False,  # Print iterations at each step
    )

    # data = Data("tickets")
    data = Data("AND_gate")
    # data = Data("OR_gate")
    data.encode_target_data()

    mlp.input_data = data.input_data
    mlp.target_data = data.target_data
    mlp.train()

    mlp.make_predictions()
    mlp.save_trained_nn()


if __name__ == "__main__":
    main()
