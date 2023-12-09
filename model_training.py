#THIS CODE IS FROM GOOGLE COLAB SINCE THE MODEL TRAINING IS DONE USING GPU GOOGLE COLAB

from google.colab import drive
drive.mount('/content/drive')
!pip install pyyaml h5py  # Required to save models in HDF5 format

from google.colab import files
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



folders = glob("C:/Users/USER/vs workspace/image_classification/try_training/*")
#print(l))

training_from_drive = '/content/drive/MyDrive/new_dataset/try_training'
testing_from_drive = '/content/drive/MyDrive/new_dataset/try_testing'
validation_from_drive = '/content/drive/MyDrive/new_dataset/try_validation'
#import the dataset

tr_data = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input, rotation_range=20, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range=[0.1,1])
#train_data_gen = tr_data.flow_from_directory(directory = r"C:\Users\USER\vs workspace\image_classification\try_training", target_size = (224,224), shuffle=False, class_mode = 'categorical' )
train_data_gen = tr_data.flow_from_directory(directory = training_from_drive, target_size = (224,224), shuffle=True, class_mode = 'categorical', batch_size = batch_size )

ts_data = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input, rotation_range=20, width_shift_range = 0.1, height_shift_range = 0.1,zoom_range=[0.1,1])
#test_data_gen = ts_data.flow_from_directory(directory = r"C:\Users\USER\vs workspace\image_classification\try_testing", target_size = (224,224), shuffle=False, class_mode = 'categorical' )
test_data_gen = ts_data.flow_from_directory(directory = testing_from_drive, target_size = (224,224), shuffle=False, class_mode = 'categorical', batch_size = batch_size )


val_data = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input, rotation_range=20, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range=[0.1,1])
#val_data_gen = val_data.flow_from_directory(directory = r"C:\Users\USER\vs workspace\image_classification\try_validation", target_size = (224,224), shuffle=False, class_mode = 'categorical' )
val_data_gen = val_data.flow_from_directory(directory = validation_from_drive, target_size = (224,224), shuffle=False, class_mode = 'categorical' , batch_size = batch_size)

images, labels = train_data_gen.next()
num_images = images.shape[0]
num_cols = 4  # Number of columns in the display grid
num_rows = np.ceil(num_images / num_cols)

fig, axes = plt.subplots(int(num_rows), num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(f'Label: {np.argmax(labels[i])}')  # Display the corresponding label
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

test= train_data_gen.batch_size
total_samples = len(train_data_gen) * batch_size
print(test)
print("Size of train_data_gen:", len(train_data_gen))



class_labels = train_data_gen.class_indices
print("Class Labels:", class_labels)

for label, index in class_labels.items():
    print("Class:", label, " - Index:", index)


#train the model

model = VGG16()
model.summary()


model =VGG16(input_shape = (224,224,3), weights = "imagenet", include_top = False)

#for layer in model.layers:
#  layer.trainable = False

for layer in model.layers[:11]:
    layer.trainable = False

opt = Adam(lr = 0.00001)
newlayers = Flatten()(model.output)
#newlayers = Dense(1024, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
#newlayers = Dense(512, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
#newlayers = Dense(128, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
#newlayers = Dense(256, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
#newlayers = Dense(512, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
#newlayers = Dense(256, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
newlayers = Dense(128, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
newlayers = Dense(64, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
newlayers = Dense(32, activation = "relu")(newlayers)
#newlayers = Dropout(0.5)(newlayers)
newlayers = Dense(num_classes, activation = "softmax") (newlayers)
new_model = Model(inputs = model.input, outputs = newlayers)
new_model.compile(loss= "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"] )
new_model.summary()
for i, layer in enumerate(new_model.layers):
    print(i, layer.name, layer.trainable)

#layers = new_model.layers
#for layer in layers:
#    if layer.weights:
#        print(layer.name)
#        for weight in layer.weights:
#            print(weight.name)
#            print(weight.shape)
#            print(weight.numpy())

#conv1_weights = new_model.layers[1].weights[0].numpy()
#print(conv1_weights)

checkpoint = ModelCheckpoint("/content/drive/My Drive/6_6.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')

#m = new_model.fit(train_data_gen, validation_data = test_data_gen, epochs = 200, steps_per_epoch = len(train_data_gen),
#                 validation_steps = len(test_data_gen),callbacks = [checkpoint, early])