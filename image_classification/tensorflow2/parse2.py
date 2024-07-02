import sys
import os
import numpy as np
import pandas as pd
name = sys.argv[1]
stri = sys.argv[2]
df = pd.read_csv(name)
df1 = df
for i, k, in enumerate(df1["Name"]):
    if k.find(stri)<0:
        df1 = df1.drop(i)
    else:
        df = df.drop(i)

df1 = df1.reset_index()
df1 = df1.drop(columns=["index"])
#name = name.replace(".csv","_GEMM.csv")
#df1.to_csv(name,index=False)
print(df1)
print(np.sum(df1["Percentage"]))
