import argparse

import src.config as config
from src.agents import basic, intermediate, advanced


def main():
    """
    Program entry point. Parses command line arguments to decide which agent to run.
    :return: None
    """
    parse_command_line_arguments()

    # Run the Basic agent to train the neural network given the data or to determine its optimal parameters.
    if config.agent == "Bas":
        basic.run_basic_agent()

    # Run the Intermediate agent to interact with the user through a text-based interface and make early predictions.
    elif config.agent == "Int":
        intermediate.run_intermediate_agent()

    # Run the Advanced agent.
    elif config.agent == "Adv":
        advanced.run_advanced_agent()

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


if __name__ == "__main__":
    main()
