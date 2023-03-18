import tensorflow as tf
import numpy as np
from random import sample
from sklearn.decomposition import PCA

class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, plt, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        plt.ion()
        
        self.losses = []
    
    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax1.clear()
        self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=self.y_test, cmap='seismic')
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(epoch), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
        self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.losses = []
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch+1, plot_loss=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
import cv2
from time import time
from tensorflow.keras.preprocessing import image as kimage

from google.colab import drive
drive.mount("/content/gdrive")


# Importing data
train_info = pd.read_csv('train_info.csv', header=None)
lista_imtrain = train_info[1].tolist()
lista_imtrain = lista_imtrain[1:len(lista_imtrain)]
print(len(lista_imtrain))

df_labels = pd.read_csv('train_info.csv',
                         names=['Class']) 
df_labels = df_labels.reset_index()
df_labels['Name'] = df_labels['level_1']
print(df_labels.shape)

train_labels = pd.DataFrame(lista_imtrain[0:len(lista_imtrain)])
train_labels.columns = ['Name']
print(train_labels.shape)

join_labels = train_labels.merge(df_labels, on='Name', how='left')
train_labels = join_labels['Class'].tolist()

annot_data = pd.read_csv("class_list.txt", sep=" ", header=None)
annot_data.columns = ["Number", "Name"]

train_labels_df = pd.DataFrame(train_labels, columns=['Number'])
join_labels = train_labels_df.merge(annot_data, on='Number', how='left')
train_labels_str = join_labels['Name'].tolist()

# Debug variable to limit the number of loaded images
val_info = pd.read_csv('val_info.csv', header=None)
lista_imvalid = val_info[1].tolist()
lista_imvalid = lista_imvalid[1:len(lista_imvalid)]
print(len(lista_imvalid))

df_labels = pd.read_csv('val_info.csv',
                         names=['Class']) 
df_labels = df_labels.reset_index()
df_labels['Name'] = df_labels['level_1']
print(df_labels.shape)

val_labels = pd.DataFrame(lista_imvalid[0:len(lista_imvalid)])
val_labels.columns = ['Name']
print(val_labels.shape)

join_labels = val_labels.merge(df_labels, on='Name', how='left')
val_labels = join_labels['Class'].tolist()

val_labels_df = pd.DataFrame(val_labels, columns=['Number'])
join_labels = val_labels_df.merge(annot_data, on='Number', how='left')
val_labels_str = join_labels['Name'].tolist()

def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    batch_holder = np.zeros((dim_train, 100, 100, 3))
    images = []
    # for image_path in lista_imtrain:
    for i in range(0,len(lista_imtrain)):
        image_path = lista_imtrain[i]
        img = kimage.load_img(path+image_path, target_size=(100, 100))
        img = kimage.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        batch_holder[i, :] = img

    return batch_holder
    
t0 = time()
x_train = imreads('/content/train_set/')
print("Time required: %0.3f seconds." % (time() - t0))

x_train.shape

def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    batch_holder = np.zeros((dim_valid, 100, 100, 3))
    images = []
    # for image_path in lista_imtrain:
    for i in range(0,len(lista_imvalid)):
        image_path = lista_imvalid[i]
        img = kimage.load_img(path+image_path, target_size=(100, 100))
        img = kimage.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        batch_holder[i, :] = img

    return batch_holder
    
t0 = time()
x_test = imreads('/content/val_set/')
print("Time required: %0.3f seconds." % (time() - t0))

x_test.shape

x_train=np.reshape(x_train,(dim_train,30000))/255.
x_test=np.reshape(x_test,(dim_valid,30000))/255.

x_test.shape

y_train = train_labels
y_test = val_labels

y_train_str = train_labels_str
y_test_str = val_labels_str


# Plot examples
def plot_triplets(examples, title = ['Anchor','Positive','Negative']):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i],(100,100,3)))
        plt.title(title[i].capitalize())
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
print(y_train[0],y_train[1],y_train[2])
print(y_train_str[0],y_train_str[1],y_train_str[2])
x_train[0].shape
plot_triplets([x_train[0], x_train[1], x_train[2]], [y_train_str[0],y_train_str[1],y_train_str[2]])


# Batch of triplets
def create_batch(batch_size=256):
    x_anchors = np.zeros((batch_size, 30000))
    x_positives = np.zeros((batch_size, 30000))
    x_negatives = np.zeros((batch_size, 30000))

    y_anchors = []
    y_positives = []
    y_negatives = []
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, len(x_train) - 1)
        y_anchors.append(y_train_str[random_index])
        x_anchor = x_train[random_index]
        y = int(y_train[random_index])

        indices_for_pos = [i for i in range(0,len(y_train)) if y_train[i] == y]
        indices_for_neg = [i for i in range(0,len(y_train)) if y_train[i] != y]
        
        index_pos = random.randint(0, len(indices_for_pos) - 1)
        y_positives.append(y_train_str[indices_for_pos[index_pos]])
        x_positive = x_train[indices_for_pos[index_pos]]

        index_neg = random.randint(0, len(indices_for_neg) - 1)
        y_negatives.append(y_train_str[indices_for_neg[index_neg]])
        x_negative = x_train[indices_for_neg[index_neg]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        
    return [x_anchors, x_positives, x_negatives]
    # , [y_anchors, y_positives, y_negatives]
    
examples = create_batch(1)
plot_triplets(examples)

def create_batch_test(batch_size=int(256/5)):
    x_anchors = np.zeros((batch_size, 30000))
    x_positives = np.zeros((batch_size, 30000))
    x_negatives = np.zeros((batch_size, 30000))

    y_anchors = []
    y_positives = []
    y_negatives = []
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, len(x_test) - 1)
        y_anchors.append(y_test_str[random_index])
        x_anchor = x_test[random_index]
        y = int(y_test[random_index])

        indices_for_pos = [i for i in range(0,len(y_test)) if y_test[i] == y]
        indices_for_neg = [i for i in range(0,len(y_test)) if y_test[i] != y]
        
        index_pos = random.randint(0, len(indices_for_pos) - 1)
        y_positives.append(y_test_str[indices_for_pos[index_pos]])
        x_positive = x_test[indices_for_pos[index_pos]]

        index_neg = random.randint(0, len(indices_for_neg) - 1)
        y_negatives.append(y_test_str[indices_for_neg[index_neg]])
        x_negative = x_test[indices_for_neg[index_neg]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        
    return [x_anchors, x_positives, x_negatives]
    
examples = create_batch(1)
plot_triplets(examples)


# Embedding model
emb_size = 251

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(30000,)),
    tf.keras.layers.Dense(251, activation='sigmoid')
])

example = np.expand_dims(x_train[0], axis=0)
example_emb = embedding_model.predict(example)[0]


# Siamese network
input_anchor = tf.keras.layers.Input(shape=(30000,))
input_positive = tf.keras.layers.Input(shape=(30000,))
input_negative = tf.keras.layers.Input(shape=(30000,))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()


# Triplet loss
alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


# Data generator
def data_generator(batch_size=256):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y

def data_generator_test(batch_size=int(256/5)):
    while True:
        x2 = create_batch_test(batch_size)
        y2 = np.zeros((batch_size, 3*emb_size))
        yield x2, y2


# Model training
batch_size = 512
epochs = 10
steps_per_epoch = int(x_train.shape[0]/batch_size)

net.compile(loss=triplet_loss, optimizer='adam')

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, 
    verbose=True,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            x_test[:500], y_test[:500]
        )]
)

net.save("siamese_net_val.h5") 


# Query
alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

from tensorflow import keras
model = keras.models.load_model('siamese_net_val.h5',
                                compile=False)

# Predict each image
prediction_list = []
names_list = []

for i in range(0,len(lista_imtrain)):
    image_path = lista_imtrain[i]
    img = kimage.load_img('/content/train_set/'+image_path, target_size=(100, 100))
    img = kimage.img_to_array(img)
    img = np.reshape(img,(1,30000))/255.

    prediction_list.append(model.predict([img, img, img], verbose=0))
    names_list.append(image_path)


# Evalutation
def getQueryLabel(img_name, train=False):
  if not train:
    df_labels = pd.read_csv('val_info.csv',
                          names=['Class'])
  else:
    df_labels = pd.read_csv('train_info.csv',
                          names=['Class'])
  df_labels = df_labels.reset_index()
  df_labels['Name'] = df_labels['level_1']
  return df_labels[df_labels['level_1']==img_name]['Class'].values[0]

prediction_list_new = []
for i in range(0,len(prediction_list)):
  prediction_list_new.append(prediction_list[i][0,:])

from sklearn.neighbors import NearestNeighbors
neighbor = NearestNeighbors(n_neighbors = 5)
neighbor.fit(prediction_list_new)

li_accuracy = []

for query_name in lista_imvalid:

  query_img = kimage.load_img('/content/val_set/' + query_name, target_size=(100, 100))
  query_img = kimage.img_to_array(query_img)
  query_img = np.reshape(query_img,(1,30000))/255.

  query_img_class = getQueryLabel(query_name)

  pred_query = model.predict([img, img, img], verbose=0)
  neighbor = NearestNeighbors(n_neighbors = 5)
  neighbor.fit(prediction_list_new)
  dist, result = neighbor.kneighbors(pred_query)
  result = result.tolist()[0]

  accuracy = 0
  for index in result:
    sim_path = lista_imtrain[index]
    sim_class = getQueryLabel(sim_path, train=True)
    if sim_class == query_img_class:
      accuracy += 1

  li_accuracy.append(accuracy/len(result))