import joblib
import sys

import numpy as np
import pandas as pd

from .common import encode_data, run_multi_layer_perceptron
import src.config as config
from src.neural_network.data_processor import append_new_ticket_to_csv, DataProcessor, inverse_encoding_no_categories
from src.neural_network.multi_layer_perceptron import MultiLayerPerceptron

# General variables.
categories = ["Credentials", "Datawarehouse", "Emergencies", "Equipment", "Networking"]
early_prediction_steps = [3, 5, 7]  # Make early predictions at these questions.


def run_intermediate_agent():
    # Initialise agent.
    _print_welcome_message()
    is_nn_retrained = False

    while True:
        # Load the re-fitted neural network with the new tickets.
        if is_nn_retrained:
            updated_csv_file_name = get_updated_csv_file_name(config.csv_file)
            data = DataProcessor(updated_csv_file_name)
            mlp_trained = joblib.load("../neural_networks/{}.joblib".format(updated_csv_file_name))
        # Load the previously trained neural network (from the basic agent).
        else:
            data = DataProcessor(config.csv_file)
            mlp_trained = joblib.load("../neural_networks/{}.joblib".format(config.csv_file))

        # Variables to reset for each new ticket.
        new_ticket = dict()
        early_prediction_made = False

        # Ask questions and store answers.
        print("\nNew ticket\nPlease answer the following questions to log a new ticket:")
        for i, tag in enumerate(data.tags):

            # Make early prediction.
            if i in early_prediction_steps:
                completed_ticket = fill_out_missing_ticket_data(data, new_ticket)
                make_prediction(mlp_trained, data, completed_ticket, is_early_prediction=True)

                # Early prediction satisfies user, stop asking questions.
                if question_yes_no("Are you happy with this prediction") == "Yes":
                    early_prediction_made = True
                    print()
                    break
                # Early prediction doesn't satisfies user, keep asking questions until next early prediction.
                else:
                    print("Please continue answering the questions below to narrow down the choices:")
                    early_prediction_made = False

            # Ask question.
            reply = question_yes_no(tag)
            new_ticket[tag] = reply

            # Exit ticket logger system when user wants to stop.
            if reply == "Exit":
                _exit_cli()

        # Make a prediction based on a full ticket (user answered all 9 questions).
        if not early_prediction_made:
            make_prediction(mlp_trained, data, new_ticket)

        # Check if user is happy with the agent's choice of response team.
        if question_yes_no("Are you happy with the response team chosen") == "No":
            desired_response_team = question_desired_team("Please choose a response team that you think suits your "
                                                          "problem best from the following choices:\n{}: "
                                                          .format(categories))
            final_ticket = new_ticket
            final_ticket["Response Team"] = desired_response_team
            new_csv_file_name = get_updated_csv_file_name(config.csv_file)
            append_new_ticket_to_csv(new_csv_file_name, final_ticket)

            # Retrain the mlp classifier.
            updated_data = DataProcessor(new_csv_file_name)
            encode_data(updated_data)
            mlp = MultiLayerPerceptron(
                name=new_csv_file_name,
                input_data=updated_data.input_data_encoded,
                target_data=updated_data.target_data_encoded,
                hidden_layers_size=mlp_trained.hidden_layer_sizes,
                solver=mlp_trained.solver,
                activation_function=mlp_trained.activation,
                learning_rate_init=mlp_trained.learning_rate_init,
                momentum=mlp_trained.momentum,
                optimisation_tolerance=mlp_trained.tol,
                num_iterations_no_change=mlp_trained.n_iter_no_change,
                max_iterations=mlp_trained.max_iter,
            )

            run_multi_layer_perceptron(mlp, is_refitted_nn=True)

            print("Your change will be taken into consideration for future tickets.")
            is_nn_retrained = True

        # Ask if user wants to submit another ticket.
        if question_yes_no("\nDo you want to submit another ticket?") == "No":
            _exit_cli()


def fill_out_missing_ticket_data(data, partial_ticket):
    most_common_values = data.input_data.mode().to_dict()
    for tag in data.tags:
        if tag not in partial_ticket:
            partial_ticket[tag] = most_common_values[tag][0]
    return partial_ticket


def make_prediction(mlp, data, new_ticket, is_early_prediction=False):
    # Convert new ticket to DataFrame and one-hot encode it for the MultiLayerPerceptron to make a prediction.
    new_ticket = pd.DataFrame(np.array(new_ticket.values()).T.tolist(),
                              index=np.array(new_ticket.keys()).T.tolist()).T
    data.input_data = new_ticket
    data.encode_input_data()

    # Make a prediction and print it to the command line.
    prediction = mlp.predict(data.input_data_encoded)
    prediction_decoded = inverse_encoding_no_categories(prediction, categories).at[0]
    probability_predictions = pd.DataFrame(mlp.predict_proba(data.input_data_encoded), columns=categories)

    if is_early_prediction:
        print("The system predicts that your ticket will be directed to \033[1m{}\033[0m".format(prediction_decoded))
    else:
        print("\nYour ticket is being directed to: \033[1m{}\033[0m\n".format(prediction_decoded))
        if config.debug:
            print("Prediction = {}\n{}\n".format(prediction, probability_predictions))


def question_yes_no(question):
    # Valid set of answers (default is "Yes").
    valid_answers = {"yes": "Yes", "y": "Yes", "yy": "Yes", "": "Yes",
                     "no": "No", "n": "No", "nn": "No",
                     "quit": "Exit", "q": "Exit", "exit": "Exit", "e": "Exit"}
    prompt = "[Yes/No]: "

    # Ask user a question and record his answer.
    sys.stdout.write("{} {}".format(question, prompt))  # Use sys.stdout.write to keep input on same line as question.
    user_answer = input().lower()

    # Map user's answer to set of valid answers.
    try:
        answer = valid_answers[user_answer]
        return answer
    except KeyError:
        print("Invalid answer entered. Please use Yes/YES/yes/y or No/NO/no/n.")
        exit(1)


def question_desired_team(question):
    while True:  # Keep asking desired team until a valid input is given.
        # Ask user a question and record his answer.
        sys.stdout.write(question)  # Use sys.stdout.write to keep input on same line as question.
        user_answer = input().title()
        if user_answer in categories:
            return user_answer
        else:
            print("The team you have chosen cannot be recognised.")


def get_updated_csv_file_name(csv_file_name):
    return "{}_updated".format(csv_file_name)


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
   | |  | |/ __| |/ / _ \ __|  | |    / _ \ / _  |/ _  |/ _ \  __|
   | |  | | (__|   <  __/ |_   | |___| (_) | (_| | (_| |  __/ |   
   |_|  |_|\___|_|\_\___|\__|  |______\___/ \__, |\__, |\___|_|   
                                             __/ | __/ |          
                                            |___/ |___/          """)
