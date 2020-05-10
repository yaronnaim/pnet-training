#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import pkg_resources
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os

FACES_PATH = '../data/face_detection/faces/'


# In[5]:


# Load the TensorBoard notebook extension
#get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime

# Clear any logs from previous runs
#get_ipython().system('rm -rf ./logs/ ')


# In[6]:


class PNet(tf.keras.Model):
    def __init__(self):
        super(PNet, self).__init__(name="PNet")
        # Define layers here.
        self.conv1 = tf.keras.layers.Conv2D(10, (3, 3), name="conv1")
        self.prelu1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2], name="prelu1")
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), name="conv2")
        self.prelu2 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2], name="prelu2")
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), name="conv3")
        self.prelu3 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2], name="prelu3")
        self.cls_output = tf.keras.layers.Conv2D(2, (1, 1), activation="softmax", name="conv4-1")
        self.bbox_pred = tf.keras.layers.Conv2D(4, (1, 1), name="conv4-2")
        #self.landmark_pred = keras.layers.Conv2D(10, (1, 1), name="conv4_3")

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        scores = None

        x = self.conv1(inputs)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        scores = [self.cls_output(x), self.bbox_pred(x)]#, self.landmark_pred(x)]
        
        return scores


# ## Dataset iterator

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))



# Set up some global variables
USE_GPU = False

if USE_GPU:
    devicedevice = '/device:GPU:0'
else:
    device = '/cpu:0'
print('Using device: ', device)


# Test the PNet  to ensure that the implementation does not crash and produces outputs of the expected shape.
# Pnet will output are:
# 1. Face classification,  size (batch,1,1,2) for 2 calss classification, "Face", and "Not face"
# 2. Bounding box  (batch,1,1,4) for 4 boundind box corrdinates (x,y,w,h)

def test_PNet(batch=64):
    model = PNet()
    with tf.device(device):
        x = tf.zeros((batch, 12, 12, 3))
        classification_scores, bbox_score = model(x)
        print(model.summary())
        print('\nP-Net output size testing: \nclassificatin score output', classification_scores.shape,
              '\nbounding box score output', bbox_score.shape)

batch_test = 32
test_PNet(batch_test)



# Read Dataset

training_size = 5000

def read_pos_images():
    #Read positive images:
    path, __, filenames = next(os.walk(FACES_PATH+'pos_train/'))
    file_count = training_size #len(filenames)
    images = np.empty([0,12,3])
    for i in range(file_count):
        j=i+1
        img=cv2.imread(f"{path}{j}.bmp")
        images=np.append(images,img,axis=0)
    #Create list of probabilities:
    prob=[]
    for i in range(file_count):
        prob.append([[[0.0,1.0]]])
    #Create list of coordinates:
    coordinates=[]
    file = open(FACES_PATH+'coordinates.txt','r')
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    idx=[1,0,3,2]
    for line in lines:
        line = line.split(" ")
        line = line[1]
        line=line[1:-1]
        line = line.split(",")
        #Transpose coordinates
        x=0
        nline=[]
        for i in idx:
            nline.append(line[i])
            x=x+1
        line=[[[float(c) for c in nline]]]
        coordinates.append(line)
    #Return images, probs, and coordinates
    return images, prob, coordinates

def read_neg_images():
    #Read negative images:
    path, __, filenames = next(os.walk(FACES_PATH+'neg_train/'))
    file_count = training_size #len(filenames)
    images = np.empty([0,12,3])
    for i in range(file_count):
        j=i+1
        img=cv2.imread(f"{path}{j}.bmp")
        images=np.append(images,img,axis=0)
    #Create list of probabilities:
    prob=[]
    for i in range(file_count):
        prob.append([[[1.0,0.0]]])
    #Create list of coordinates:
    coordinates=[]
    for i in range(file_count):
        coordinates.append([[[0.0,0.0,0.0,0.0]]])
    #Return images, prob, coordinates
    return images, prob, coordinates

#Read in all images, probabilities, and coordinates
pimages, pprob, pcoordinates = read_pos_images()
nimages, nprob, ncoordinates = read_neg_images()
o_images=np.append(pimages,nimages,axis=0)
o_images=np.reshape(o_images,(-1,12,12,3))
o_prob=pprob+nprob
o_coordinates=pcoordinates+ncoordinates

#Shuffle them up using an index
idx=np.arange(len(o_prob))
np.random.shuffle(idx)
images=np.empty_like(o_images)
c=0
for i in idx:
    images[c]=o_images[i]
    c=c+1
#images=(np.float32)(images-127.5)/128.0
images=(np.float32)(images)/255

#images = np.transpose(images, (0, 2, 1, 3)) #Transpose images
prob=[]
for i in idx:
    prob.append(o_prob[i])
coordinates=[]
for i in idx:
    coordinates.append(o_coordinates[i])

print('X_train , Image batch shape ', images.shape)
print('y_train , Classification ground true batch shape ' ,np.array(prob).shape)
print('y_train , Coordinates ground true batch shape ', np.array(coordinates).shape)


X_data = images
del(images)
y_data = np.concatenate((np.array(prob), np.array(coordinates)), axis=3)

# ## Divide dataset to "train', "val" and "test"
def load_data(X, y, training_prec = 0.7, val_prec = 0.1, test_prec = 0.2):
        data_length = len(X)
        num_training = np.int(data_length * training_prec)
        num_validation = np.int(data_length * val_prec)
        
        mask = range(num_training)
        X_train = X[mask]
        y_train = y[mask]
        mask = range(num_training, num_training + num_validation)
        X_val = X[mask]
        y_val = y[mask]
        mask = range(num_training + num_validation, data_length)
        X_test = X[mask]
        y_test = y[mask]
        
        return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_data(X_data, y_data)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)



def train(model_init_fn, optimizer_init_fn, num_epochs=1, is_training=False):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on  training set and periodically checks
    accuracy on the validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Nothing, but prints progress during trainingn
    """    
    with tf.device(device):
        
        #Set up summary writers to write the summaries to disk in a different logs directory:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        #compute the loss function over the classification and ovr bounding box 
        classification_loss = tf.keras.losses.BinaryCrossentropy()
        bbox_loss = tf.keras.losses.MeanSquaredError()        
        
        model = model_init_fn()
        optimizer = optimizer_init_fn()
        
        train_loss = tf.keras.metrics.BinaryCrossentropy(name='train_classification_loss')
        train_bbox_loss = tf.keras.metrics.MeanSquaredError(name='train_bbox_loss')
        
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            
        #val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_loss = tf.keras.metrics.BinaryCrossentropy(name='val_classification_loss')
        val_bbox_loss = tf.keras.metrics.MeanSquaredError(name='val_bbox_loss')

        val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
        
        t = 0
        for epoch in range(num_epochs):
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_bbox_loss.reset_states()
            
            train_accuracy.reset_states()
            
            for x_np, y_np in train_dset:
                with tf.GradientTape() as tape:
                    
                    # Use the model function to build the forward pass.
                    classification_scores, bbox_scores = model(x_np, training=True)
                    prediction_loss = classification_loss(y_np[:,:,:,:2], classification_scores)
                    coordinate_loss = bbox_loss(y_np[:,:,:,2:], bbox_scores)
                    loss = prediction_loss + 0.5 * coordinate_loss * y_np[:,:,:,1]
                    # Print loss 
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
                    # Update the metrics
                    train_loss.update_state(y_np[:,:,:,:2], classification_scores)
                    train_bbox_loss.update_state(y_np[:,:,:,2:], bbox_scores*y_np[:,:,:,1] )
                    train_accuracy.update_state(y_np[:,:,:,:2], classification_scores)
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_bbox_loss.reset_states()
                        val_accuracy.reset_states()
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            classification_scores, bbox_scores = model(test_x, training=False)
                            t_prediction_loss = classification_loss(test_y[:,:,:,:2], classification_scores)
                            t_coordinate_loss = bbox_loss(test_y[:,:,:,2:], bbox_scores)
                            t_loss = t_prediction_loss + 0.5 * t_coordinate_loss * test_y[:,:,:,1]

                            val_loss.update_state(test_y[:,:,:,:2], classification_scores)
                            val_bbox_loss.update_state(test_y[:,:,:,2:], bbox_scores*test_y[:,:,:,1])
                            val_accuracy.update_state(test_y[:,:,:,:2], classification_scores)
                            
                            with test_summary_writer.as_default():
                                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

                        
                        template = 'Iteration {}, Epoch {}, \nLoss: {}, Bbox loss: {}, Accuracy: {},\nVal Loss: {}, Val Bbox Loss: {}, Val Accuracy: {}'
                        print (template.format(t, epoch+1,
                                             train_loss.result(),
                                             train_bbox_loss.result(),
                                             train_accuracy.result()*100,
                                             val_loss.result(),
                                             val_bbox_loss.result(),  
                                             val_accuracy.result()*100))
                    t += 1
    return model


print_every = 10
num_epochs = 150

def model_init_fn():
    return PNet()

def optimizer_init_fn():
    learning_rate = 1e-3
    return tf.keras.optimizers.Adam(learning_rate) 
    #return tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    
model = train(model_init_fn, optimizer_init_fn, num_epochs=num_epochs, is_training=True)

# Test data
predictions = model.predict(X_test)


import matplotlib.pyplot as plt
import pandas as pd

score = predictions[0]
bbox = predictions[1]

score = np.squeeze(score)
bbox = np.squeeze(score)

y_test_score = np.squeeze(y_test[:,:,:,:2])
y_test_bbox = np.squeeze(y_test[:,:,:,2:])

from sklearn.metrics import confusion_matrix, classification_report


print(classification_report(y_test_score, np.round(score)))
print(confusion_matrix(y_test_score[:,1:2], np.round(score[:,1:2])))





