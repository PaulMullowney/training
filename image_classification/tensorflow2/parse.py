import sys
import os
import numpy as np
f = sys.argv[1]
with open(f) as fp:
    lines = fp.readlines()

ulines, counts = np.unique(lines, return_counts=True)
print(len(lines),len(ulines))
i = np.argsort(counts)
counts = counts[i]
ulines = ulines[i]
f = open(sys.argv[2],"w")
for ul, c in zip(ulines,counts):
    print(c, ":", ul)    
    f.write("%s"%ul)
f.close()
