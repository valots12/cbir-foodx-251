# Imports
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from time import time
import tensorflow as tf
from tensorflow import keras
from skimage import io 
from keras import Model
from keras import Input
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# KDTree creation and saving
from sklearn.neighbors import KDTree
import joblib

from google.colab import drive
drive.mount("/content/gdrive")


# Import train features and labels
train_info = pd.read_csv('train_labels.csv', header=None)
lista_imtrain = train_info[1].tolist()
lista_imtrain = lista_imtrain[1:len(lista_imtrain)]

train_mobilenet = np.load('train_features.npy')
train_mobilenet.shape

df_labels = pd.read_csv('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Annot/train_info.csv',
                         names=['Class']) 
df_labels = df_labels.reset_index()
df_labels['Name'] = df_labels['level_1']

train_labels = pd.DataFrame(lista_imtrain[0:len(lista_imtrain)])
train_labels.columns = ['Name']

join_labels = train_labels.merge(df_labels, on='Name', how='left')
train_labels = join_labels['Class'].tolist()
train_labels = join_labels['Class']

one_hot_encoded_train_labels = tf.keras.utils.to_categorical(train_labels)

similarity_df = pd.DataFrame(lista_imtrain, columns=['Name'])
similarity_df['Label'] = train_labels
similarity_df['Similarity'] = np.nan

val_info = pd.read_csv('val_labels.csv', header=None)
lista_imvalid = val_info[1].tolist()
lista_imvalid = lista_imvalid[1:len(lista_imvalid)]


# Definition model
input_t = Input(shape=(224, 224, 3))
temp = MobileNetV2(weights='imagenet', input_tensor=input_t)

desired_layer = temp.get_layer('global_average_pooling2d')
newmodel = Model(inputs=input_t, outputs=desired_layer.output)

def mobilenet_features(img):
    x = kimage.img_to_array(img)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    f = newmodel.predict(x, verbose=False)
    return f

# Get feature vector of an image by given model and img_path
def getFeatureVector(img_name):

  # Load file and extract features
  img_path = '/content/val_set/'+img_name
  image = kimage.load_img(img_path, target_size=(224, 224))
  features = mobilenet_features(image)

  features = np.array(features)
  return features

# Define cosine similarity between feature vectors A and B 
def getCosineSimilarity(A, B):
  cos_similarity = np.dot(A,B.T) / (np.linalg.norm(A)*np.linalg.norm(B)) 
  return cos_similarity

def getQueryLabel(img_name, train=False):
  if not train:
    df_labels = pd.read_csv('val_labels.csv',
                          names=['Class'])
  else:
    df_labels = pd.read_csv('train_labels.csv',
                          names=['Class'])
  df_labels = df_labels.reset_index()
  df_labels['Name'] = df_labels['level_1']
  return df_labels[df_labels['level_1']==img_name]['Class'].values[0]

annot_data = pd.read_csv("annot_list.txt", sep=" ", header=None)
annot_data.columns = ["Number", "Name"]

# Plot similar 5 images with given image and similar images dataframe
def plotSimilarImages(img_file, similar_df):
  img = kimage.load_img('/content/val_set/' + img_file, target_size=(224, 224))

  img_class = getQueryLabel(img_file)
  fig, axarr = plt.subplots(2,3,figsize=(15, 8))
  axarr[0,0].imshow(img)
  axarr[0,0].set_title("TEST IMAGE - " + "\nClass: " + annot_data[annot_data['Number']==img_class]['Name'].values[0])
  axarr[0,0].axis('off')

  j, k, m = 0, 0, 1
  for index, sim in similar_df.iterrows():
    sim_class = getQueryLabel(sim['Name'], train=True)
    similarity = sim['Similarity']

    similar = kimage.load_img('Data/Train/' + sim['Name'], target_size=(224, 224))
    axarr[k,m].imshow(similar)
    axarr[k,m].set_title("Similarity: %.3f" % similarity + "\nClass: " + annot_data[annot_data['Number']==sim_class]['Name'].values[0])
    axarr[k,m].axis('off')

    m += 1
    if m == 3 and k != 1:
      k += 1
      m = 0

    j += 1
    if j == 5:
      break

  plt.tight_layout()
  plt.show()

def evaluate(sim_df, query_class):
  accuracy = 0
  for i in sim_df['Label'][0:5].values:
    if i == query_class:
      accuracy += 1
  return accuracy/5

# Get and plot 5 similar images for given image path and features dataframe
def getSimilarImages(img_file, evaluation=False):
  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getCosineSimilarity(img_features, train_mobilenet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=False)

  if evaluation:
    # return sorted_df, getQueryLabel(img_file)
    return evaluate(sorted_df, getQueryLabel(img_file))

  plotSimilarImages(img_file, sorted_df.head(5))


# Load file and extract features
img_path = 'Data/Validation/' + 'val_000008.jpg'
image = kimage.load_img(img_path, target_size=(224, 224))
plt.imshow(image)

# Get similar images of test image
t0 = time.time()
getSimilarImages(lista_imvalid[5])
print("Time required: %0.3f seconds." % (time.time() - t0))


## Test con KDTree
# tree = KDTree(train_mobilenet)

# # Saving the search tree
# joblib.dump(tree, 'cbri_kdtree.joblib')

# name = 'val_000003.jpg'

# # Loading a query image
# query_image = kimage.load_img('Data/Validation/'+name, target_size=(224, 224))
# plt.imshow(query_image)

# # Computing query features
# query_features = mobilenet_features(query_image)
# query_features = np.array(query_features)
# query_features.shape

# # Search
# dist, ind = tree.query(query_features, k=10)

# def getQueryLabel_train(img_name):
  
#   df_labels = pd.read_csv('train_labels.csv',
#                           names=['Class'])
#   df_labels = df_labels.reset_index()
#   df_labels['Name'] = df_labels['level_1']
#   return df_labels[df_labels['level_1']==img_name]['Class'].values[0]

# target_class = getQueryLabel(name)
# accuracy = 0

# for i in range(0,ind.shape[1]):
#   if getQueryLabel_train(lista_imtrain[ind[0][i]]) == target_class:
#     accuracy += 1
# print(accuracy/ind.shape[1])

# img = query_image
# img_class = getQueryLabel('val_000003.jpg')
# fig, axarr = plt.subplots(2,3,figsize=(15, 8))
# axarr[0,0].imshow(img)
# axarr[0,0].set_title("TEST IMAGE - " + "\nClass: " + annot_data[annot_data['Number']==img_class]['Name'].values[0])
# axarr[0,0].axis('off')

# j, k, m = 0, 0, 1
# for i in range(0,5):

#   name_similar = lista_imtrain[ind[0][i]]
#   img = kimage.load_img('/content/train_set/' + name_similar, target_size=(224, 224))
#   img_class = getQueryLabel(name_similar, train=True)
#   axarr[k,m].imshow(img)
#   axarr[k,m].set_title("Similarity: %.3f" % dist[0][i] + "\nClass: " + annot_data[annot_data['Number']==img_class]['Name'].values[0])
#   axarr[k,m].axis('off')

#   m += 1
#   if m == 3 and k != 1:
#     k += 1
#     m = 0

#   j += 1
#   if j == 5:
#     break

# plt.tight_layout()
# plt.show()