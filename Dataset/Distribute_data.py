# Import libraries
import numpy as np 
import pandas as pd
import h5py

# Read the dataset for example h5 file
f = h5py.File('LetterColorImages_123.h5', 'r')
# List all groups
keys = list(f.keys())


# Create tensors (i.e. X) and targets (i.e. Y)
backgrounds = np.array(f[keys[0]])
tensors = np.array(f[keys[1]])
targets = np.array(f[keys[2]])
print ('Tensor shape:', tensors.shape)
print ('Target shape', targets.shape)
print ('Background shape:', backgrounds.shape)

# Save training and test tensors of each target separately for each user, (80,20) --> (Train, Test)
for i in set(targets):
    data=tensors[np.where(targets==i)]
    nP=len(data)
    idx_p = np.random.permutation(nP)
    training_idx_p, test_idx_p = idx_p[:int(nP*0.8)], idx_p[int(nP*0.8):]
    train = data[training_idx_p]
    test = data[test_idx_p]
    np.savez('user_%d'%(i), Xtr=train, Xte= test, label=np.repeat(i, len(tensors[np.where(targets==i)],)))