import cv2                
import numpy as np        
import os                  
from random import shuffle 

path='train_data'

IMG_SIZE = 96

def create_train_data():
    training_data = []
    label=0
    for (dirpath,dirnames,filenames) in os.walk(path,topdown=True):
      
        for dirname in dirnames:
            for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
                for file in files:
                    actual_path=path+"/"+dirname+"/"+file
                    img=cv2.imread(actual_path,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                    training_data.append([np.array(img),label])
            label=label+1           
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
train_data = create_train_data()


