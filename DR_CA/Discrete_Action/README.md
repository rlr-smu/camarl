# Shaped Reward Model for Credit Assignment (Discrete Action)
This repository contains code base for chapter 6, Shaped Reward Model for Credit Assignment.

## Installations

Our code is implemented and tested on python 3.6 and rest of the python packages can be installed using 

<code> pip install -r req.txt </code>

## Running experiments

All the parameters for the project are provided in python file <code>parameters.py</code> . The map parameters for synthetic data is at the location <code>./route_james</code>,  and baseline algorithms are specified by AGENT_NAME.

For **DIFF_RW**, select  <code>AGENT_NAME = "diff_mid_hyp"</code>

For **MTMF**, select <code>AGENT_NAME = "mtmf_sm_tgt_lg"</code>

For **MCAC** baseline, select  <code>AGENT_NAME = "global_count"</code>

For **LOCAL_DR** baseline, select  <code>AGENT_NAME = "appx_dr_colby"</code>

For **AT-DR** baseline, select  <code>AGENT_NAME = "appx_dr_scot"</code>

For **AT-BASELINE** baseline, select  <code>AGENT_NAME = "ppo_bl"</code>

Run the command: <code> python bs.py</code>


