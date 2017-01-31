The purpose of this project was (a) to construct supervised and unsupervised learning models for predicting animal features (given input data) and assess the performances of each, as well as (b) to evaluate how low-level/concrete features (i.e., size) and high-level/abstract features (i.e., intelligence) would affect each model.
Python is used for the implementation code. The following packages should be imported: pandas, numpy, scipy, sys, and sklearn, from which the KMeans function was imported for the unsupervised learning model. 


SUPERVISED:
Our supervised learning model was fed a set of testing animals to binarily categorize according to size [big (1) or small (0)], as well intelligence, [high (1) or low (0)]. This applied supervised approach constitutes a “nearest prototype classifier design” model, in which classification of the observations are determined according to which observations fall more closely to which centroid. Here we’ve implemented the multivariate Gaussian prototype approach in hopes that this model could possibly capture the input feature cases accurately; however, other models could have been implemented instead.


UNSUPERVISED:
Our unsupervised learning model separated the testing animal set into two distinct groups based on how the animals resembled each other in terms of their features, without having any specific classification goal in mind. Implementation for our unsupervised algorithm was modeled after Macqueen’s (1967) K-means algorithm, a simple unsupervised learning method for clustering data.  Conceptually within this method, k centroids (corresponding to k clusters) are placed into the space represented by the objects being clustered.  The algorithm iteratively assigns each object in the space to the closest centroid until the centroids can no longer be updated in space (i.e., no longer move). 


We selected size and intelligence, specifically, as our predictor categories because we were interested in how information could be derived from features at a concrete level (i.e., size) in comparison to at an abstract level (i.e., intelligence).  These two features were extracted from a larger data set of human ratings on 77 different animals across four designated features (i.e., size, intelligence, fierceness, and speed) (Holyoak et al., 1981).  Through choosing size to represent concrete feature types and, oppositely, intelligence to represent abstract feature types, we were hopeful in gaining insight towards whether any interaction existed when these levels of processing differed for the supervised versus unsupervised models.
 


classificationScript.py is the supervised learning algorithm that reads in either “sizetrainingdata” or “intelligencetrainingdata” (each data set is a scaled-down, consolidated data set deriving from  utilizing multivariate Gaussian distributions).

clusteringScript.py is an unsupervised learning algorithm that reads in 