import numpy as np

def purity (crosstab_, size_data):
    max_col = np.max(crosstab_, axis=0)
    sum_col = np.sum(crosstab_, axis=0)
    prt_score = purity_score(max=max_col, sum=sum_col)
    resuil = np.sum((sum_col/size_data)*prt_score, axis=0)
    
    return resuil

def purity_score (max, sum):
    return max/sum