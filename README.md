# Randomized COMMIT
Repository for the paper: Randomly COMMITting: Iterative Convex Optimization for Microstructure-Informed Tractography

The repo contains the model code and weights used to produce the results for the network classifier in the paper. 



## Abstract

Tractography is extensively utilized in brain connectivity studies using diffusion magnetic resonance imaging (dMRI) data. However, the presence of anatomically implausible and redundant streamlines is a significant challenge. Several tractogram filtering methods have been developed to eliminate false-positive streamlines and address these issues. This study introduces a tractography filtering method - Randomized COMMIT (rCOMMIT) - that is based on the Convex Optimization Modeling for Microstructure Informed Tractography (COMMIT) filtering method. The method aims to mitigate the biases of COMMIT for individual streamlines by assessing each streamline in multiple tractogram compositions to estimate an acceptance rate per streamline. In order to reduce the computational cost, this acceptance rate is used to create pseudo-labels that are used to train neural network classifiers in a semi-supervised manner. Specifically, we train a 1D-convolutional network on streamline characteristics, achieving an area under the receiver operating characteristic curve (AUC ROC) of approximately 90 % in distinguishing between plausible and non-plausible streamlines. The results from rCOMMIT are compared with those from randomized SIFT, and the intersections between the two methods are analyzed in relation to the streamline acceptance agreement. 
