# Shaped Reward Model for Credit Assignment (Continuous Action)
This repository contains code base for chapter 6, Shaped Reward Model for Credit Assignment.

## Installations

Our code is implemented and tested on python 3.6 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . The map parameters for synthetic data is at the location <code>./route_james</code>,  and baseline algorithms are specified by AGENT_NAME.

For **DR_Cont**, select  <code>AGENT_NAME = "diff_mid_hyp"</code>

Run the command: <code> python bs.py</code>

## Experimental Parameters

|             Parameters             |        Values         |
| :--------------------------------: | :-------------------: |
|           Learning Rate            |         1e-3          |
|              Horizon               |          200          |
|             Optimizer              |         Adam          |
|              Discount              |         0.99          |
| Number of hidden layers per sector |           2           |
|  Number of hidden units per layer  |          30           |
|           Non-linearity            |      Leaky Relu       |
|         Replay buffer size         |          100          |
|           Default action           | No speed change index |

