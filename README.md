# Neural-Network-Ticketing-Routing-Agent [![HitCount](http://hits.dwyl.io/Adamouization/Neural-Network-Ticketing-Routing-Agent.svg)](http://hits.dwyl.io/Adamouization/Neural-Network-Ticketing-Routing-Agent) [![GitHub license](https://img.shields.io/github/license/Adamouization/Neural-Network-Ticketing-Routing-Agent)](https://github.com/Adamouization/Neural-Network-Ticketing-Routing-Agent/blob/master/LICENSE)

**Neural-Network-Ticketing-Routing-Agent** is a neural-network-based ticketing routing agent. The agent is trained and tested with a multilayer feedforward neural network, and interacts with a user through a command-line interface, allowing the agent to ask the user questions to create a new ticket, with the capacity to make early predictions and retrain if the user comes up with a new combination of answers for a ticket. The optimal parameters are found with a grid search algorithm that tests 12,600 unique combinations of parameters (over 5 runs for even 80%/20% data splits), narrowing the neural network down to 14 optimal combinations. The agent is developed in Python 3.7 using the [Scikit-Learn](https://scikit-learn.org), NumPy, Pandas and Matplotlib libraries.

The report, which includes a summary of features implemented, design & implementation decisions (data encoding and training/testing split), evaluation (training/testing result visualisation in plots and heatmaps, grid search algorithm for determining optimal hyperparameters) and testing sections, can be read [here](https://github.com/Adamouization/Neural-Network-Ticketing-Routing-Agent/blob/master/report/report.pdf).

## Usage

Clone the repository (or download the zipped project):
`$ git clone https://github.com/Adamouization/Neural-Network-Ticketing-Routing-Agent`

Create a virtual environment for the project and activate it:

```
virtualenv ~/Environments/Neural-Network-Ticketing-Routing-Agent
source Neural-Network-Ticketing-Routing-Agent/bin/activate
```

Once you have the virtualenv activated and set up, `cd` into the project directory and install the requirements needed to run the app:

```
pip install -r requirements.txt
```

You can now run the app:
```
python A4Main.py [-h] -a AGENT -c CSV [-g] [-d]
```

where:

* `AGENT` is the type of agent to run: `[Bas, Int, Adv]`:
    * `Bas`: Train and test the neural network with the optimal parameters, or run the Grid Search algorithm to determine the optimal parameters.
    * `Int`: CLI text-based application to submit a new ticket and predict to which response team it should go.
    * `Adv`: Train and test a decision tree classiﬁer.
* `CSV` is the CSV file containing the data used to train/test the data.
* `-g`: flag set to run the grid search algorithm.
* `-d`: flag set to enter debug mode, printing more statements to the command line.
* `-h`: flag for help on how to use the agent (prints instructions on the command line).

Examples:
* `python A4Main.py -a Bas -c tickets -d` to train/test the neural network.
* `python A4Main.py -a Bas -c tickets -g` to run the grid search algorithm.
* `python A4Main.py -a Int -c tickets` to submit a new ticket through the CLI text-based interface.
* `python A4Main.py -a Adv -c tickets` to train/test the decision tree.
* `python A4Main.py -h` for help on how to run the agent.

## Contact
* Email: adam@jaamour.com
* Website: www.adam.jaamour.com
* LinkedIn: [@adamjaamour](https://www.linkedin.com/in/adamjaamour/)
* Twitter: [@Adamouization](https://twitter.com/Adamouization)
