# Tensor Dictionary Learning with representation quantization for Remote Sensing Observation Compression

This repository contains MATLAB codes and scrips designed for the compression of tensor data based on a novel
tensor dictionary learning method that uses the CANDECOMP/PARAFAC (CP) decomposition, as it is presented 
in the paper "Tensor Dictionary Learning with representation quantization for Remote Sensing Observation 
Compression" (A. Aidini, G. Tsagkatakis, P. Tsakalides). In the proposed method, a dictionary of specially 
structured tensors is estimated using an Alternating Direction Method of Multipliers (ADMM) approach, as 
well a symbol encoding dictionary is learned from training samples. Given the learned models, a new sample 
is first presented by a set of sparse coefficients corresponding to linear combination of the elements of 
the learned dictionary. Then, the derived coefficients are quantized and encoded in order to be transmitted, 
significantly reducing the number of bits required to represent the useful information of the data.

## Requirements

### Dataset
The efficacy of the proposed compression algorithm is evaluated over time series of satellite-derived observations 
and more specifically time series of normalized difference vegetation index (NDVI). In more detail, the files 
training_data.mat and testing_data.mat contain 50 training and 40 testing times series of size 200 x 200 x 7 of our
experiments, where the last dimension indicates the number of days.

### Tensor Toolbox
We use the tensor toolbox for MATLAB, which is available in https://www.tensortoolbox.org and contains useful 
functions for tensor operators.

## Contents
**demo_tdlc.m** : The primary script that loads the data, performs the compression using the proposed tensor dictionary 
learning method in combination with quantization and encoding of the tensor data, and provides the results.

**Sthresh.m** : Perform the hard-thresholding operator.

**outprod.m** : Compute the outer product of two vectors.

**Unfold.m** : Unfold the tensor into a matricization mode.

**Fold.m** : Fold a matricization mode into a tensor.

## References
1. A.  Aidini,  G.  Tsagkatakis,  and P.  Tsakalides, “Tensor Dictionary Learning with representation quantization 
for Remote Sensing Observation Compression,” in Proc. 2020 Data Compression Conference, Snowbird, UT,  March 24-27, 2020.

2. Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox Version 3.1, Available online, June 2019.
URL: https://gitlab.com/tensors/tensor_toolbox.
