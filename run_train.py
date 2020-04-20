from train import RAN
import os

NumUsers=33
identities={}

if __name__ == '__main__':
    for i in range(1,NumUsers+1):
    	identities[i]='user_%d.npz'%(i)
        print('{} user number {}'.format(identities[i].split('.')[0],i)) 

        
        if not os.path.exists('SavedModel/%s'%identities[i].split('.')[0]):
            os.makedirs('SavedModel/%s'%identities[i].split('.')[0])
        if not os.path.exists('ReconstructedImages/%s'%identities[i].split('.')[0]):
            os.makedirs('ReconstructedImages/%s'%identities[i].split('.')[0])        
 
        
        # Train the reconstructive adversarial network (RAN)
        ran = RAN(identities[i].split('.')[0])
        ran.train(epochs=30000, batch_size=256, save_interval=50)

