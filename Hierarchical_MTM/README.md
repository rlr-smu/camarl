# Hierarchical Learning Approach for Maritime Traffic Management
This repository contains code base for chapter 5, Hierarchical Learning Approach for Maritime Traffic Management.

## Installations

Our code is implemented and tested on python 3.6 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . The map parameters for synthetic data is at the location <code>./synData/lucas</code>,  and baseline algorithms are specified by AGENT_NAME.

For **IMVF-PG** baseline, select  <code>AGENT_NAME = "op_ind"</code>

For **VESSEL-PG**, select <code>AGENT_NAME = "pg_fict"</code>

For **META-PG** baseline, select  <code>AGENT_NAME = "op_pgv"</code>

Run the command: <code> python launch.py</code>



## Experimental Parameters

|            Parameters            | Values |
| :------------------------------: | :----: |
|          Learning Rate           |  1e-3  |
|        Delay Penalty (Wd)        |   1    |
|             Horizon              |  100   |
|            Optimizer             |  Adam  |
|             Discount             |  0.99  |
|      Total number of zones       |   30   |
| Number of hidden layers per zone |   2    |
| Number of hidden units per layer |   30   |
|          Non-linearity           |  tanh  |
|        Replay buffer size        |  200   |
|      Number of meta-action       |   4    |