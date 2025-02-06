import pandas as pd
import numpy as np

all_signals=pd.read_csv("train_test.csv")

types=np.unique(all_signals['Material Type'])

for mat in types:
    print(mat, len(all_signals[all_signals['Material Type']==mat]))
