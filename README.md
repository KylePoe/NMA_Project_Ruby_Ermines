# NMA Project - Ruby Ermines
Group project for neuromatch academy 2020, ruby ermines pod.

## Project Proposal

### Scientific question:
What can subsampling tell us about the amount of neurons required to accurately predict the dimensionality of the total dataset?

### Brief scientific background:
Variability within V1 is not confined to visual stimuli, but heavily dependent on animal state. The latent variables of this state are important for interpreting V1 circuit activity, but the amount of neurons necessary to record from to characterize this activity is not well known.

### Proposed analysis:
We will randomly sample neurons in populations of m size and run several analyses for a range of m. We will compute the dimensionality of the data using PCA and thresholds on cumulative variance explained. We may also use GPFA, IF, etc., time permitting. The dimensionality will be given a confidence interval based on bootstrapping or k-fold sampling. This will give a distribution of dimensionality as a function of sample size. We will then attempt to answer the question of whether the dimension of the entire population dynamics can be predicted from the dimensionality of a subset of the population at a certain threshold of certainty. Further investigations will gauge the relationship of data dimensionality with different measures of connectivity, such as noise correlation, covariance, and cross-correlation.

### Predictions:
If the number of dimensions reaches a saturation point at a certain number of neurons, this could show redundancies across V1. On the other hand, it may not be possible to gauge the dimensionality of the entire population, which would justify the fine-grained spatial measurements obtained through Ca2+ imaging. 

### Dataset:
Stringer
