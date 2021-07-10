import numpy as np

def purity (crosstab_, size_data):
    max_col = np.max(crosstab_, axis=0) # compute max value of each group 
    sum_col = np.sum(crosstab_, axis=0) # compute sum value of each group
    prt_score = max_col/sum_col
    result = np.sum((sum_col/size_data)*prt_score, axis=0)
    
    return result
