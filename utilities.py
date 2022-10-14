#A function that convert and hdf5 dataset to a numpy dataset 
import h5py
import sys
import numpy as np

def load_data():
    hf = h5py.File('/home/ichiro19/Desktop/GITREPO/Image-Recognition-Dogs-vs-Cats/datasets/trainset.hdf5', 'r')
    X_train = np.array(hf['/X_train'])
    Y_train = np.array(hf['/Y_train'])
    
    hf = h5py.File('/home/ichiro19/Desktop/GITREPO/Image-Recognition-Dogs-vs-Cats/datasets/testset.hdf5', 'r')
    X_test = np.array(hf['/X_test'])
    Y_test = np.array(hf['/Y_test'])

    return X_train, Y_train, X_test, Y_test


    
def main():
    X_train, Y_train, X_test, Y_test = load_data()
    print(X_train, Y_train, X_test, Y_test)
   


if __name__ == "__main__":
    sys.exit(main())     



