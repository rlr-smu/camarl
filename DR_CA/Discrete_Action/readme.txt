

# Simulation Interval:
-----------------------

> You need to change two parameters - plugin python file(update_interval) and .cfg file(simdt)

air speed = 156 ( 241 in visual)
-------------------
sim_interval | t
-------------------
12 | 5
30 | 13
60 | 3

> You want your simulator to be as responsive as possible, e.g if you take an action at time t you want your simulator to respond at t itself, but in this example aircraft itself take some time to reach that speed(acceleration time).

# Speed Visual
---------------
> There is some difference in the speed you apply in simulator to the one you see in the visuals.

speed(apply) | speed(visual)
-----------------------------
156 | 241
346 | 329