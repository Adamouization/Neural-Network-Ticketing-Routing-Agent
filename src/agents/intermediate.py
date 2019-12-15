import sys

import src.config as config


def run_intermediate_agent():
    _print_welcome_message()
    new_ticket = dict()
    questions = ["Request", "Incident", "WebServices", "Login", "Wireless", "Printing", "IdCards", "Staff", "Student"]
    for q in questions:
        reply = question_yes_no(q)
        new_ticket[q] = reply
    print(new_ticket)


def question_yes_no(question):
    # Valid set of answers (default is "Yes").
    valid_answers = {"yes": "Yes", "y": "Yes", "YES": "Yes", "Yes": "Yes", "": "Yes",
                     "no": "No", "n": "No", "NO": "No", "No": "No"}
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
    print(""" _   _                 _______ _      _        _       \n| \ | |               |__   __(_)    | |      | |    _ \n|  \| | _____      __    | |   _  ___| | _____| |_  (_)\n| . ` |/ _ \ \ /\ / /    | |  | |/ __| |/ / _ \ __|    \n| |\  |  __/\ V  V /     | |  | | (__|   <  __/ |_   _ \n|_| \_|\___| \_/\_/      |_|  |_|\___|_|\_\___|\__| (_)""")
    print()
    print("Note: default answer is: 'Yes'.")
    print()
