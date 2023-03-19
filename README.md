# Content Based Image Retrieval using FoodX-251 dataset

***Part of Visual Image Processing project | UniMiB***

The purpose of the project is to compare three different methods for retrieving the best 5 similar images given a query one.

<img src="Images/query_example.jpg" width=640>

Main differences between the methods:

1. CNN as feature extractor using cosine similarity

<img src="Images/cnn_architecture.jpg" width=320 height=120>

2. BoWA (Bag of Words using AKAZE descriptors)

<img src="Images/akaze_example.jpg" width=320 height=120>

Reference: *Muhammad, Usman, et al. "Bag of words KAZE (BoWK) with two‐step classification for high‐resolution remote sensing images." IET Computer Vision 13.4 (2019): 395-403.*

3. Siamese network trained using triplet loss as feature extractor

<img src="Images/triplet_example.jpg" width=320 height=120>
