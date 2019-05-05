import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils,print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as K

image_x=50
image_y=50
'''
Reading the CSV file obtained from the images
'''
data=pd.read_csv("train.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,1:]
Y=Y[:,0]

'''
Normalization and Train Test Split
'''

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)

##for i in range(X_train.shape[0]):
##    for j in range(X_train.shape[1]):
##        X_train[i][j]=X_train[i][j]/255
##for i in range(X_test.shape[0]):
##    for j in range(X_test.shape[1]):
##        X_test[i][j]=X_test[i][j]/255
##Y_train=Y_train/255
##Y_test=Y_test/255

##X_test=X_test/255
##
##Y_train=Y_train/255
##
##Y_test=Y_test/255

'''
One Hot Encoding
'''

train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(train_y.shape[0],train_y.shape[1])
test_y=test_y.reshape(test_y.shape[0],test_y.shape[1])
X_train=X_train.reshape(X_train.shape[0],50,50,1)
X_test=X_test.reshape(X_test.shape[0],50,50,1)
X_train=X_train/255
X_test=X_test/255
train_y=train_y/255
test_y=test_y/255

'''
keras machine learning model
'''


def keras_model(image_x,image_y):
    num_of_classes=10
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

    model.add(Conv2D(64,(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    
    model.add(Conv2D(64,(5,5),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes,activation='softmax'))

    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    filepath='handEmo.h5'
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]

    return model,callbacks_list

model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=10,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("Error : ",(100-scores[1]*100))
print_summary(model)

model.save('handEmo.h5')
