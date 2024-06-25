# Experimental Test Script
import torch as t
import numpy as np
import pandas as pd


sets = ["active", "over", "part_bwd", "part_fwd", "vary_reg", "vary_rnd", "shift_fwd", "shift_bwd"]
data = pd.DataFrame()
model = t.load_model() #something like that
# Test 1 - Active frames

for set in sets:
    results = [] # store all results
    sequences = [] # get all test sequences in set
    labels = []
    for label,sequence in zip(labels,sequences):
        out = model.run(sequence)
        res = (out == label)
        results.append(res)
