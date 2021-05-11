import numpy as np

clo = [[0,1],[1,0]]
clx = [[0,0],[1,1]]
for pt in clo + clx:
    t = 2*int(pt in clo) - 1
    print(t)