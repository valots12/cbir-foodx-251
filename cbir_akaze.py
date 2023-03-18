# Imports
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from time import time
from tensorflow.keras.preprocessing import image as kimage

from google.colab import drive
drive.mount("/content/gdrive")


def imreads(path):
    """
    This reads all the images in a given folder and returns the results
    """
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        # print(image_path)
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images

dictionary_size = 1024

# Loading images
imgs_data = []

# imreads returns a list of all images in that directory
t0 = time()
imgs = imreads('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Train/')
print("Time required: %0.3f seconds." % (time() - t0))

# AKAZE Descriptors
for i in range(len(imgs)):
    # create a numpy to hold the histogram for each image
    imgs_data.insert(i, np.zeros((dictionary_size, 1)))

def get_descriptors(img, detector):
    # returns descriptors of an image
    return detector.detectAndCompute(img, None)[1]

# Extracting descriptors
detector = cv2.AKAZE_create(threshold = 0.01)

t0 = time()
desc = np.array([])
# desc_src_img is a list which says which image a descriptor belongs to
desc_src_img = []
for i in range(len(imgs)):
    img = imgs[i]
    descriptors = get_descriptors(img, detector)
    if len(desc) == 0:
        desc = np.array(descriptors)
        for j in range(len(descriptors)):
          desc_src_img.append(i)
    elif descriptors is not None:
        desc = np.vstack((desc, descriptors))
    # Keep track of which image a descriptor belongs to
        for j in range(len(descriptors)):
          desc_src_img.append(i)
# important, cv2.kmeans only accepts type32 descriptors
desc = np.float32(desc)
print("Time required: %0.3f seconds." % (time() - t0))


# Clustering
dictionary_size = 251

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 251)
kmeans.fit(desc)


# Build BoWA (Bag of Words AKAZE)
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def build_BoW(path):

    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    list_histograms = []
    list_names = []

    for image_path in images_path:
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)

        descriptors = get_descriptors(img, detector)

        if descriptors is not None:
          descriptors = np.float32(descriptors)
          histogram = build_histogram(descriptors, kmeans)
          list_histograms.append(histogram)
          list_names.append(image_path)

    return list_histograms, list_names

t0 = time()
bow_akaze, bow_names = build_BoW('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Train/')
print("Time required: %0.3f seconds." % (time() - t0))


# Query
query_name = 'val_000001.jpg'
query_img = cv2.imread('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Validation/' + query_name, flags=cv2.IMREAD_GRAYSCALE)
query_desc = get_descriptors(query_img, detector)
query_desc = np.float32(query_desc)

if (query_desc is not None):
    histogram = build_histogram(query_desc, kmeans)

from sklearn.neighbors import NearestNeighbors
neighbor = NearestNeighbors(n_neighbors = 5)
neighbor.fit(bow_akaze)
dist, result = neighbor.kneighbors([histogram])

annot_data = pd.read_csv("/content/gdrive/MyDrive/Progetto FoodX-251/Data/Annot/class_list.txt", sep=" ", header=None)
annot_data.columns = ["Number", "Name"]

def getQueryLabel(img_name, train=False):
  if not train:
    df_labels = pd.read_csv('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Annot/val_info.csv',
                          names=['Class'])
  else:
    df_labels = pd.read_csv('/content/gdrive/MyDrive/Progetto FoodX-251/Data/Annot/train_info.csv',
                          names=['Class'])
  df_labels = df_labels.reset_index()
  df_labels['Name'] = df_labels['level_1']
  return df_labels[df_labels['level_1']==img_name]['Class'].values[0]

# Plot similar 5 images with given image and similar images dataframe
def plotSimilarImages(img_file, list_similar_indexes, list_similar_distances):

  img = kimage.load_img('/content/val_set/' + img_file, target_size=(224, 224))
  img_class = getQueryLabel(img_file)
  fig, axarr = plt.subplots(2,3,figsize=(15, 8))
  axarr[0,0].imshow(img)
  axarr[0,0].set_title("TEST IMAGE - " + "\nClass: " + annot_data[annot_data['Number']==img_class]['Name'].values[0])
  axarr[0,0].axis('off')

  j, k, m = 0, 0, 1
  for i in range(0,len(list_similar_indexes)):

    sim_index = list_similar_indexes[i]
    sim_path = bow_names[sim_index]
    sim_class = getQueryLabel(sim_path.split('Train/')[1], train=True)
    similarity = list_similar_distances[i]

    similar = kimage.load_img(sim_path, target_size=(224, 224))
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

  plotSimilarImages(query_name, result.tolist()[0], dist.tolist()[0])