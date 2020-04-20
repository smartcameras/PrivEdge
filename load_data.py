import scipy
from glob import glob
import numpy as np
import scipy.misc

class DataLoader():
    def __init__(self, img_res=(32, 32)):
        self.img_res = img_res
    
    def load_data(self,dataset_name,batch_size=1, is_testing=False):
        data_type = "Xtr" if not is_testing else "Xte" 
        path = 'Dataset/%s.npz' %(dataset_name)
        data = np.load(path)
        traintest = data[data_type]
        if is_testing:
           batch_size=len(traintest)
        idx = np.random.choice((len(traintest)), size=batch_size)
        imgs= []
        for i in idx:
            img = traintest[i]
            imgs.append(img)

        imgs = np.array(imgs) / 127.5 - 1.

        return imgs

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)



