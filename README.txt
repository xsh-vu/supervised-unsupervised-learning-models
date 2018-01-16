The purpose of this project was to:

	(a) construct supervised and unsupervised cognitive learning models for predicting 	animal features (given input data) and assess the performances of each model, as 	well as 

	(b) evaluate how low-level/concrete features (i.e., size) and high-level/abstract 	features (i.e., intelligence) would affect each model.

Implementation coded in Python. The following packages should be imported: pandas, numpy, scipy, sys, and sklearn (from which the KMeans function was imported for the unsupervised learning model). 


SUPERVISED:
The supervised learning model was fed a set of testing animals to binarily categorize according to size [big (1) or small (0)], as well intelligence, [high (1) or low (0)]. This applied supervised approach constitutes a “nearest prototype classifier design” model, in which classification of the observations are determined according to which observations fall more closely to which centroid. Here, the multivariate Gaussian prototype approach was implemented in hopes that this model could possibly capture the input feature cases accurately; however, other models could have been implemented instead.


UNSUPERVISED:
The unsupervised learning model separated the testing animal set into two distinct groups based on how the animals resembled each other in terms of their features, without having any specific classification goal in mind. Implementation for this unsupervised algorithm was modeled after Macqueen’s (1967) K-means algorithm, a simple unsupervised learning method for clustering data. Conceptually within this method, k centroids (corresponding to k clusters) are placed into the space represented by the objects being clustered. The algorithm iteratively assigns each object in the space to the closest centroid until the centroids can no longer be updated in space (i.e., no longer move). 


Size and intelligence, specifically, were selected as predictor categories in order to see how information could be derived from features at a concrete level (i.e., size) in comparison to at an abstract level (i.e., intelligence). These two features were extracted from a larger data set of human ratings on 77 different animals across four designated features (i.e., size, intelligence, fierceness, and speed) (Holyoak et al., 1981).  Through choosing size to represent concrete feature types and, oppositely, intelligence to represent abstract feature types, I hoped to gain insight towards whether any interaction existed when these levels of processing differed for the supervised versus unsupervised models.
 


	- classificationScript.py is the supervised learning algorithm script.

	- clusteringScript.py is the unsupervised learning algorithm script. 


For more info on procedures, data, and results, see: https://docs.google.com/document/d/1GrJiSx_I6sDgiKSHaW1SibZm7dt6m90rKJ39J7yJ8O4/edit?usp=sharing