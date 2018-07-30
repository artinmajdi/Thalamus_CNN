import sys


for input in sys.argv:
    if input.split('=')[0] == 'nuclei':
        A = input.split('=')[1]

        if A[0] == '[':
            B = A.split('[')[1].split(']')[0]
            B = B.split(",")
            B = [int(k) for k in B]
            print(B)

import numpy as np
a = range(4,14)
a = np.append([1,2,4567],a)
a
