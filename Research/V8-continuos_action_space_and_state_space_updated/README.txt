V8.1
trying to sample actions into discreet buckets of 0%,25%,50%,75%,100% allocation

V8.2
actions are randomly sampled using uniform distribution to get any decimal value from 0 to 1

V8.3
same as V8.2 but with multiprocessing of training to alleviate the problem of starting point dependence

V8.4
logging multiple runs of training to see the starting point dependence

V8.5
nifty with multiprocessing and multiple features

V8.6
V8.5 with also experimenting with action filter (>0.8 means invest 100%, <0.2 means 0%)

Most stable version for sinx - 8.3
Most stable version for nifty - 8.6