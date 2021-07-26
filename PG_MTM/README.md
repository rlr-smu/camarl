# Policy Gradient Approach for Maritime Traffic Management
This repository contains code base for chapter 4, Policy Gradient Approach for Maritime Traffic Management.

## Installations

Our code is implemented and tested on python 2.7 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project is provided in python file <code>parameters.py</code> . The synthetic data experiments are run on the map parameter <code>MAP_ID = "11_2"</code>,  and baseline algorithms are specified by AGENT_NAME.

For **VESSEL-PG** agent in select <code>AGENT_NAME = "pg_fict"</code>

For **PG** agent in select  <code>AGENT_NAME = "pg_vanilla"</code>

For **DDPG** agent in select  <code>AGENT_NAME = "ddpg"</code>

For **MAX SPEED** agent in select  <code>AGENT_NAME = "tmin"</code>

Now you can run the command:

<code> python main.py</code>

