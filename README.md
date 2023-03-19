# Content Based Image Retrieval using FoodX-251 dataset

***Part of Visual Image Processing project | UniMiB***

The purpose of the project is to compare three different methods for retrieving the best 5 similar images given a query one.

<img src="Images/query_example.jpg" width=640>
<br/><br/>
Main differences between the methods:
<br/><br/>
**1. Transfer learning from pre-trained CNN**

- Extraction feature from the first dense layer of MobileNet v2 (1x1280)
- Cosine similarity as similarity measure
- For-loop structure to check all possible combinations of pairs of images
- Tentative with KDTree to reduce complexity of the searching algorithm (from O(n) to O(logn)) but lower performance

<img src="Images/cnn_architecture.jpg" width=320 height=120>
<br/><br/>
**2. BoWA (Bag of Words using AKAZE descriptors)**

- Identification of most relevant keypoint of each image using 0.01 as threshold for AKZE descriptors
- Application of K-means clustering algorithm to detect centroids (K = 251)
- Representation of each image according to the obtained centroids (BOW approach)
- KNN as similarity measure

<img src="Images/akaze_example.jpg" width=320 height=120>

Reference: Muhammad, Usman, et al. "Bag of words KAZE (BoWK) with two‐step classification for high‐resolution remote sensing images." IET Computer Vision 13.4 (2019): 395-403.
<br/><br/>
**3. Siamese network trained using triplet loss as feature extractor**

- Definition of embedding model and replication of it using triplet loss
- Each triplet is composed of an Anchor, a Positive example and a Negative one
- Use of the trained NN as feature extractor
- KNN as similarity measure

<img src="Images/triplet_example.jpg" width=320 height=120>
