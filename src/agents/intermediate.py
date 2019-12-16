import joblib
import sys

import numpy as np
import pandas as pd

import src.config as config
from src.neural_network.data_processor import DataProcessor, inverse_encoding_no_categories
from src.neural_network.multi_layer_perceptron import MultiLayerPerceptron

questions = ["Request", "Incident", "WebServices", "Login", "Wireless", "Printing", "IdCards", "Staff", "Student"]
categories = ["Credentials", "Datawarehouse", "Emergencies", "Equipment", "Networking"]


def run_intermediate_agent():
    # Initialise agent.
    _print_welcome_message()

    while True:
        # Ask questions and store answers.
        print("\nNew ticket\nPlease answer the following questions to log a new ticket:")
        new_ticket = dict()
        for q in questions:
            reply = question_yes_no(q)
            new_ticket[q] = reply
            if reply == "Exit":
                _exit_cli()

        # Convert new ticket to DataFrame and one-hot encode it for the MultiLayerPerceptron to make a prediction.
        new_ticket = pd.DataFrame(np.array(new_ticket.values()).T.tolist(),
                                  index=np.array(new_ticket.keys()).T.tolist()).T
        data = DataProcessor(config.csv_file)
        data.input_data = new_ticket
        data.encode_input_data()

        # Load the previously trained neural network (from the basic agent).
        mlp = joblib.load("../neural_networks/{}.joblib".format(config.csv_file))

        # mlp = MultiLayerPerceptron(
        #     name=config.csv_file,
        #     input_data=data.input_data_encoded,
        #     target_data=None,
        #     hidden_layers_size=mlp_temp.hidden_layer_sizes,
        #     solver=mlp_temp.solver,
        #     activation_function=mlp_temp.activation,
        #     learning_rate_init=mlp_temp.learning_rate_init,
        #     momentum=mlp_temp.momentum,
        #     optimisation_tolerance=mlp_temp.tol,
        #     num_iterations_no_change=mlp_temp.n_iter_no_change,
        #     max_iterations=mlp_temp.max_iter,
        # )

        # Make a prediction and print it to the command line.
        prediction = mlp.predict(data.input_data_encoded)
        prediction_decoded = inverse_encoding_no_categories(prediction, categories).at[0]
        probability_predictions = pd.DataFrame(mlp.predict_proba(data.input_data_encoded), columns=categories)
        print("\nYour ticket is being directed to: \033[1m{}\033[0m\n".format(prediction_decoded))

        if config.debug:
            print("Prediction = {}\n{}\n".format(prediction, probability_predictions))

        if question_yes_no("Do you want to submit another ticket?") == "No":
            _exit_cli()


def question_yes_no(question):
    # Valid set of answers (default is "Yes").
    valid_answers = {"yes": "Yes", "y": "Yes", "YES": "Yes", "Yes": "Yes", "yy": "Yes", "": "Yes",
                     "no": "No", "n": "No", "NO": "No", "No": "No", "nn": "No",
                     "Quit": "Exit", "QUIT": "Exit", "quit": "Exit", "q": "Exit", "Exit": "Exit", "EXIT": "Exit",
                     "exit": "Exit", "e": "Exit"}
    prompt = "[Yes/No]: "

    # Ask user a question and record his answer.
    sys.stdout.write("{} {}".format(question, prompt))
    user_answer = input()

    # Map user's answer to set of valid answers.
    try:
        answer = valid_answers[user_answer]
        return answer
    except KeyError:
        print("Invalid answer entered. Please use Yes/YES/yes/y or No/NO/no/n.")
        exit(1)


def _print_welcome_message():
    _print_ascii_title()
    print()
    print("You can quit at any time by typing 'exit' or 'quit'.")
    print("Note: default answer is: 'Yes'.")


def _exit_cli():
    print("\nExiting Ticket Logger.")
    exit(0)


def _print_ascii_title():
    print(""" _______ _      _        _      _                                 
|__   __(_)    | |      | |    | |                                
   | |   _  ___| | _____| |_   | |     ___   __ _  __ _  ___ _ __ 
   | |  | |/ __| |/ / _ \ __|  | |    / _ \ / _` |/ _` |/ _ \ '__|
   | |  | | (__|   <  __/ |_   | |___| (_) | (_| | (_| |  __/ |   
   |_|  |_|\___|_|\_\___|\__|  |______\___/ \__, |\__, |\___|_|   
                                             __/ | __/ |          
                                            |___/ |___/          """)
