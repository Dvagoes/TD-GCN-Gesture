# Experimental Test Script
import torch as t
import numpy as np
import pandas as pd

datasets = ["active", "over", "part_bwd", "part_fwd", "vary_reg", "vary_rnd", "shift_fwd", "shift_bwd"]
data = pd.DataFrame()
model = t.load_model() #something like that
# Test 1 - Active frames

for ds in datasets:
    root = "data/DHG14-28/gen_dhgdataset_" + ds + ".py"
    results = [] # store all results
    sequences = [] # get all test sequences in set
    labels = [] # 
    for label,sequence in zip(labels,sequences):
        out = model.run(sequence)
        res = (out == label)
        results.append(res)
