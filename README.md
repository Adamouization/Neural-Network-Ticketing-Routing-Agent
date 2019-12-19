# Neural-Network-Ticketing-Routing-Agent

**Neural-Network-Ticketing-Routing-Agent** is a neural-network-based ticketing routing agent.

The agent is developed in Python 3.7 using [SciKit](https://scikit-learn.org)'s library.

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
    * `Adv`: todo.
* `CSV` is the CSV file containing the data used to train/test the data.
* `-g`: flag set to run the grid search algorithm.
* `-d`: flag set to enter debug mode, printing more statements to the command line.
* `-h`: flag for help on how to use the agent.

Examples:
* `python A4Main.py --agent Bas --csv tickets --debug`
* `python A4Main.py --agent Bas --csv tickets --gridsearch`
* `python A4Main.py --agent Int --csv tickets`
* `python A4Main.py -h`

## Contact
* Email: adam@jaamour.com
* Website: www.adam.jaamour.com
* LinkedIn: [@adamjaamour](https://www.linkedin.com/in/adamjaamour/)
* Twitter: [@Adamouization](https://twitter.com/Adamouization)
