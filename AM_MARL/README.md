# Credit Assignment for Sparse Reward Model
This repository contains code base for the chapter credit assignment for sparse reward model.

## Installations

Our code is implemented and tested on python 3.6 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . Baseline algorithms are specified by AGENT_NAME.

Run the command: <code> python main.py</code>

## Experimental Parameters

|            Parameters            | Values |
| :------------------------------: | :----: |
|          Learning Rate           |  1e-3  |
|             Horizon              |  100   |
|            Optimizer             |  Adam  |
|             Discount             |  0.99  |
| Number of hidden layers per grid |   2    |
| Number of hidden units per layer |   32   |
|          Non-linearity           |  tanh  |
|        Replay buffer size        |  400   |

