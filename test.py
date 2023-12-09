\
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Lambda, Input, Dropout
from keras.models import Model, load_model
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


num_classes = 7
batch_size = 32
IMAGE_SHAPE = [224,224]
epoch = 10


#image prediction
threshold = 0.5
model = load_model(r"C:\Users\USER\vs workspace\image_classification\FINALWEIGHTS.h5")
img = tf.keras.utils.load_img(r"C:\Users\USER\Downloads\didi6.jpg", target_size = (224,224))
#img.show()
print(img.size)

x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

class_labels = ['didi','jojo','nana','didi and jojo', 'didi and nana','jojo and nana', 'all']
predict_image = model.predict(x)
print(str(predict_image)+" the initial")

highest_index = np.argmax(predict_image,axis = -1)
print(str(highest_index)+ " the highest")

highest_value = np.max(predict_image)
print("the prediction : "+str(predict_image)+" the highest value"+str(highest_value) )

if highest_value > threshold:
    print(class_labels[highest_index[0]])

else:
    print("i dont know")

example = model.predict(np.array(x))
print(str(example)+" example je")
print(predict_image[0][1])
preddd = predict_image.flatten()
print(preddd)