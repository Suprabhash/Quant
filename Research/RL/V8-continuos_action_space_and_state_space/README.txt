V8.1
trying to sample actions into discreet buckets of 0%,25%,50%,75%,100% allocation

V8.2
actions are randomly sampled using uniform distribution to get any decimal value from 0 to 1

V8.3
same as V8.2 but with multiprocessing of training to alleviate the problem of starting point dependence

V8.4
logging multiple runs of training to see the starting point dependence

V8.5
fixing eval_q_learning bug