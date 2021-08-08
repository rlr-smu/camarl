import os





hpath = os.getenv("HOME")



cmd = "--sim --detached --scenfile count.scn " + " --config-file count_sac.cfg"

os.system(hpath+"/miniconda3/envs/bluesky/bin/python BlueSky.py " + cmd)

