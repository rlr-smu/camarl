# Shaped Reward Model for Credit Assignment (Cooperative-Navigation)
This repository contains code base for chapter 6, Shaped Reward Model for Credit Assignment.

## Installations

Our code is implemented and tested on python 3.6 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . Baseline algorithms are specified by AGENT_NAME. 

For **DIFF_RW**, select  <code>AGENT_NAME = "diff_mid"</code>

For **MF**, select <code>AGENT_NAME = "mean_field"</code>

For **LOCAL_DR** baseline, select  <code>AGENT_NAME = "appx_dr_colby"</code>

Run the command: <code> python main.py</code>


