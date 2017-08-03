import numpy as np

def minibatch(data, labels, size = 100):
  '''Return a tuple of of size=size from a predictor data
     and labels data '''
  indexes = np.arange(0, len(labels))
  np.random.shuffle(indexes)
  batch_indexes = indexes[:size]
  X = [data[i] for i in batch_indexes]
  Y = [labels[i] for i  in batch_indexes]
  
  return np.asarray(X), np.asarray(Y)

 
