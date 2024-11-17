#Predicting using saved model.
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
import numpy as np

model = tf.keras.models.load_model('C:\Users\username\Model',
                                   custom_objects=None,
                                   compile=True)
testpath = 'C:\Users\username\Data\Test_data'
files = os.listdir(testpath)
for idx, file in enumerate(files):
    test = img_to_array(load_img(testpath+file))
    test = resize(test, (224,224), anti_aliasing=True)
    test*= 1.0/255
    lab = rgb2lab(test)
    l = lab[:,:,0]
    L = gray2rgb(l)
    L = L.reshape((1,224,224,3))
    #print(L.shape)
    vggpred = newmodel.predict(L)
    ab = model.predict(vggpred)
    #print(ab.shape)
    ab = ab*128
    cur = np.zeros((224, 224, 3))
    cur[:,:,0] = l
    cur[:,:,1:] = ab
    plt.imshow(lab2rgb(cur))
   