from scipy.misc import imread
import numpy as np
import pandas as pd
import os

root='./hello'

for directory,subdirectories,files in os.walk(root):
    for file in files:
        print(file)
        im=imread(os.path.join(directory,file))
        value=im.flatten()
        value=np.hstack((directory[8:],value))
        df=pd.DataFrame(value).T
        df=df.sample(frac=1)
        with open("train.csv","a") as dataset:
            df.to_csv(dataset,header=False,index=False)




