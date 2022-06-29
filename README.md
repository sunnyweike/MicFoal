# MicFoal

MicFoal - Multiclass Imbalanced and Concept Drift Network Traffic Classification Framework Based on Online Active Learning - which includes a configurable supervised learner for the initialization of a network traffic classification model, an active learning method with a hybrid label request strategy, a label sliding window group, a sample training weight formula and an adaptive adjustment mechanism for the label cost budget based on periodic performance evaluation. In addition, a novel uncertain label request strategy based on a variable least confidence threshold vector is designed to address the problem of the multiclass imbalance ratio changing or even the number of classes changing over time. Experiments performed based on 8 well-known real network traffic datasets demonstrate that MicFoal is more effective and efficient than several state-of-the-art learning algorithms.

For more details about MicFoal and CALMID , please check out our papers [1, 2].

[1] Weike Liu, Zhaoyun Ding, Hang Zhang, Qingbao Liu, Cheng Zhu. *Multiclass Imbalanced and Concept Drift Network Traffic Classification Framework Based on Online Active Learning*

[2] Weike Liu, Hang Zhang, Zhaoyun Ding, Qingbao Liu, Cheng Zhu.  *A comprehensive active learning method for multiclass imbalanced data streams with concept drift*, Knowledge-Based Systems, Volume 215, 5 March 2021.

## 

## Authors

- **Weike Liu** - [homepage](https://www.researchgate.net/profile/Weike-Liu)

  

## Source code

- MicFoal.java



## Experimental environment

- moa-release-2019.05
-  The default parameter settings are ***sizeWindow*** = 500 and ***b*** = 0.2.



## Setup - Parameter sensitivity experiments

The compared parameters include the size of the sliding label window, ***sizeWindow***, and the labelling cost budget, ***b***. 

### Data

Number of data stream: 1

- Entry10

### Parameter settings

-  The value of ***b*** falls with in {0.15, 0.2, 0.25, 0.3}
- The value of ***sizeWindow*** falls with in {300, 500, 700, 1000}

### Metrics

- AvFβ (β=1)
- CBA
- mGM
- CEN

### Experimental evaluation results

1. Details: The detailed evaluation results of the parameter (***sizeWindow*** **,** ***b***) sensitivity experiments based on Entry10
2. Mean: Average results of 10 repeated experiments
3. Std: Standard deviation results of 10 repeated experiments



## Setup - Component dedication analysis experiments

The purpose of the experiments is to analyse the contribution of the following components to our MicFoal method: the initial supervised learner, hybrid label request strategy (H), sample training weight formula (S) and adaptive adjustment mechanism of the label cost budget based on periodic performance evaluation (P). 

### Data

Number of data stream: 1

- Entry10

### Methods

MicFoal with different component settings: 
-  2 initial supervised learners:
  - Adaptive Random Forests (ARF)
  - Leveraging Bagging (LB)
- The hybrid strategy proposed in this paper and 3 traditional label request strategies:
  - Hybrid label request strategy (H)
  - The uncertainty strategy with a variable threshold (VU)
  - The uncertainty strategy with a random and variable threshold (RV)
  - The uncertainty strategy with selective sampling (SS)
- Sample training weight formula (S)
- Adaptive adjustment mechanism of the label cost budget based on periodic performance evaluation (P)

### Metrics

- AvFβ (β=1)
- CBA
- mGM
- CEN

### Experimental evaluation results 

1. Details: The detailed evaluation results of the comparison experiments with different configurations of components of MicFoal
2. Mean: Average results of 10 repeated experiments
3. Std: Standard deviation results of 10 repeated experiments



## Setup - Comparative experiments

We compare MicFoal with other state-of-the-art methods on the 8 real-world network traffic streams. 

### Data

Number of data streams: 8

- Entry10
- KDDTrain5
- UDP_NoPorts
- TCP_NoPorts
- NIMS1
- URL_All
- URL_BestFirst
- URL_Infogain

### Methods

Number of methods: 4

- MicFoal 
- ARFre
- CALMID
- RAL

### Metrics

- Recall
- Precision
- AvFβ (β=1)
- CBA
- mGM
- CEN

### Experimental evaluation results 

1. Details: The detailed evaluation results of experiment recorded after repeated 10 comparative experiments
2. Mean: Average results of 10 repeated experiments
3. Std: Standard deviation results of 10 repeated experiments



