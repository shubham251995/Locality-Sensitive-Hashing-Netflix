# Locality Sensitive Hashing for User Similarity

## Overview
This project implements Locality Sensitive Hashing (LSH) to identify similar users within the Netflix Challenge dataset. It aims to find pairs of users who rate movies similarly using three different similarity measures: Jaccard similarity, cosine similarity, and discrete-cosine similarity.

## Dataset
The dataset is a cleaned version of the original Netflix Challenge dataset, containing records of users who rated between 300 and 3000 movies. The dataset includes 65,225,506 records with user IDs, movie IDs, and ratings, renumbered to eliminate gaps.

## Methods
### Signature Matrix Creation
- **Sparse Matrix Conversion**: Converts dataset to a Compressed Sparse Column (CSC) matrix.
- **Binary Conversion**: For Jaccard similarity, ratings are converted to binary values.
- **Min-hashing and Random Permutations**: Used to generate signature matrices for Jaccard and cosine similarities, respectively.

### Locality-Sensitive Hashing (LSH)
- **Banding Technique**: Signature matrix is split into bands to hash users into buckets, enhancing the chance of finding similar user pairs.
- **Candidate Pair Generation**: From the buckets, potential similar user pairs are generated for further analysis.

### Similarity Measures
- **Jaccard Similarity**: Measures similarity based on the intersection over union of rated movies.
- **Cosine Similarity**: Uses the cosine of the angle between rating vectors.
- **Discrete-Cosine Similarity**: Similar to cosine similarity but uses binary rating vectors.

## Results
- **Efficiency of Different Band Numbers**: Explored how different numbers of bands affect the performance and number of similar user pairs identified.
- **Time Constraints**: Experiments were capped at 20 minutes to balance between computational feasibility and depth of analysis.
