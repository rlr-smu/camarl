# Multiagent Decision Making For Maritime Traffic Management
This repository contains the code for paper Multiagent Decision Making For Maritime Traffic Management. AAAI-19

## Installations

Our code is implemented and tested on python 2.7 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . The synthetic data experiments are run on the map parameter <code>MAP_ID = "11_2"</code>,  and baseline algorithms are specified by AGENT_NAME.

For **VESSEL-PG**, select <code>AGENT_NAME = "pg_fict"</code>

For **PG** baseline, select  <code>AGENT_NAME = "pg_vanilla"</code>

For **DDPG** baseline, select  <code>AGENT_NAME = "ddpg"</code>

For **MAX SPEED** baseline, select  <code>AGENT_NAME = "tmin"</code>

Run the command: <code> python main.py</code>

## Experimental Parameters

|            Parameters            | Values |
| :------------------------------: | :----: |
|          Learning Rate           |  1e-3  |
|        Delay Penalty (Wd)        |   1    |
|             Horizon              |  200   |
|            Optimizer             |  Adam  |
|             Discount             |  0.99  |
|      Total number of zones       |   23   |
| Number of hidden layers per zone |   2    |
| Number of hidden units per layer |   23   |
|          Non-linearity           |  tanh  |
|        Replay buffer size        |  400   |
