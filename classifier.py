import numpy as np
def score_function(weight, bias, data):
    print(data)
    w_t = np.transpose(weight)
    w_tx = np.matmul(w_t, data)
    return w_tx + bias
   

def trainer(file):
    
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            clean_data = line.strip()
            use_data = clean_data.split(',')
            w = np.zeros(len(use_data))
            print(len(use_data))
            score1 = score_function(w, 0, clean_data)
            score2 = score_function(w, 0, clean_data)
            score3 = score_function(w, 0, clean_data)
    return score1, score2, score3
