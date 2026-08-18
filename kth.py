import numpy as np
import pandas as pd
import kthsup as sup
import networkx as nx
import math

# Makes printing weighted graph more legiable
np.set_printoptions(precision=3, suppress=True, threshold=np.inf)

n = 3*5+1
k= 3


path = "kth"+str(k)+","+str(n)+".json"
# df = sup.search_optimal(n,k, epochs1=20, epochs2=10**6, rate=2e-3)
# OR (auto is likely better/faster)
#df = sup.auto_search_optimal(n, k, epochs1=10, epochs2=10**6, rate=1e-2, norm_tol=.3, min_val=math.floor(n/3)-1 )
df = pd.read_json(path)
print(len(df))
#Print the first weighted and flattened matrices in the date set.
#print( np.asarray(df['W'].iloc[0]) )
#print( np.asarray(df['A'].iloc[0]) )