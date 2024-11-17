from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
import numpy as np

#Define local path
path = 'C:\Users\username\Data'

#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)                     #for ReLu

train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode=None)

"""
VGG16 is expecting an image of 3 dimension with size 224x224 as an input, hence in preprocessing we have to scale all 
images to 224 instead of 256.
"""



#Convert from RGB to Lab
"""
Here we are basically iterating on each image, we convert the RGB to Lab. 
Think of LAB image as a grey image in L channel and all color information stored in A and B channels as explained earlier.
The input to the network will be the L channel, so we assign L channel to X vector. 
And assign A and B to Y.
"""

X =[]                                                    #converting images to arrays
Y =[]
for img in train[0]:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128)                        #A and B values range from -127 to 128, 
                                                         #so we divide the values by 128 to restrict values to between -1 and 1.
  except:
     print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))                              #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)


#now we have one channel of L in each layer but, VGG16 is expecting 3 dimension, 
#so we repeated the L channel two times to get 3 dimensions of the same L channel

vggfeatures = []
for i, sample in enumerate(X):
  sample = gray2rgb(sample)
  sample = sample.reshape((1,224,224,3))
  prediction = newmodel.predict(sample)
  prediction = prediction.reshape((7,7,512))
  vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

