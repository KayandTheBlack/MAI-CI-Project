# MAI-CI-Project
A Reinforcemenet learning based NEAT approach on OpenAI Gym's Car Racing game

Each folder contains one experiment: 
  - baseline considers a population of 32 individuals, FeedForward NN, speciation threshold of 3.0, elitism of 2, and 0 units in the initial hidden layer
  - Elitism: baseline with elitism = 16
  - Hidden 20 units: baseline with hidden layer initialized with 20 units
  - Recurrent: baseline allowing Recurent neural networks
  - Speciation: baseline with specieation threshold of 1.0 
  - Population 100: baseline with population of 100 individuals
  - Population 100 RNN: Population 100 allowing Recurent neural networks
  - Population 100 Speciation: Population 100 with speciation threshold set to 1.0
  - Population 100 RNN Speciation: Population 100 with speciation threshold set to 1.0 and allowing recurrent neural networks.

A Reuirements file is placed inside each folder

To execute eny experiment, install the requirements with
pip install -r requirements.txt
and run the code with:
python name_of_file.py
(for example to run baseline: python baseline.py)
