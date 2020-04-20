from keras_contrib.layers.normalization import InstanceNormalization
from keras.models import load_model
import os
import numpy as np
from load_data import DataLoader


def loss_dec_test_BC(g, data_test, data_test_gt, size):
    diffs=[]
    for i in range(1,size):
        rec_imgs = g[i].predict(data_test)
        rec_imgs = np.reshape(rec_imgs,(rec_imgs.shape[0],32*32*3))
        data_test_c = np.reshape(data_test,(data_test.shape[0],32*32*3))
        diff = np.mean(np.square(rec_imgs- data_test_c), axis=-1)
        diffs.append(diff)
    return (diffs)


data_loader = DataLoader(img_res=(32, 32))

identities={}
g={}
NUM=34

# Load one-class classifiers of all users
for i in range(1,NUM):
	identities[i]='user_%d.npz'%(i)
	g[i] = load_model('SavedModels/%s/%s.h5'%(identities[i].split('.')[0],identities[i].split('.')[0]), compile=False)
	print('The model of {} is loaded'.format(identities[i]))


# Prediction 
positives=[]
for i in range(1,NUM):
        identity=identities[i]
        print(i,identity)
        
        #Load and preprocess all of the test images of user i_th
        imgs = data_loader.load_data(identity=identity.split('.')[0], is_testing=True)
        imgs=np.asanyarray(imgs)
                
        # Measure the dissimilarity between original images and its N corrosponding reconstructions
        error= loss_dec_test_BC( g,  imgs, imgs, NUM)
        error=np.asarray(error)
        positive=[]
        
        #-------------
        #(1)Calculate the minimum similarity for each images of user i,
        #(2)Predicted label= label of one-class classifier with minimum dissimilarity
        #(3)Report TP, FN, FP and TN
        #------------
        error_min = error.argmin(0)
        for j in range(NUM-1):
            positive.append((error_min==j).sum())
        positives.append(positive)
        print(positive) 
cm=np.asarray(positives,dtype=np.float32)
TP=np.diag(cm)
FP = np.sum(cm,axis=0) - TP
FN = np.sum(cm,axis=1) - TP
num_classes = NUM-1
TN=[]
for i in range(num_classes):
    temp = np.delete(cm,i,0)
    temp = np.delete(temp,i,1)
    TN.append(sum(sum(temp)))
recall = TP/(TP+FN)
precision = TP/(TP+FP)
print('Per user recall:\n {},\n per user precision is \n {}'.format(recall, precision))
recall = np.mean(recall)
precision = np.mean(precision)
print(' recall is\n {}, precission is\n {}' .format(recall, precision))

