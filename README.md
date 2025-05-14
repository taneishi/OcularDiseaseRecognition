# Classification of Ocular Diseases based on Fundus Images

## Introduction

According to the statistics in 2020 of WHO, globally the number of visually impaired people is estimated to be 285 million, of whom 39 million are blind.
Visual impairment including blindness is a major global health issue because it significantly reduce QoL.

On the other hand, about 80% of blindness is believed to be due to preventable causes, and one of the major causes are ocular diseases.
Ocular diseases such as cataract, glaucoma, and AMD can lead to blindness if they progress. For example, the first cause of blindness is cataract, accounting for 51%.
There is a significant need for early diagnosis and screening of ocular diseases in order to prevent visual impairment leading to blindness.

Here we introduce ocular disease recognition model that uses eye fundus images as input and trains multi-labels of ocular diseases in a supervised learning.
The results of inference are evaluated by *area under the curve*, AUC of *receiver operating characteristics*, ROC.

## Dataset

The *ODIR-2019* dataset includes both eye fundus images of 3,353 patients for training[^ODIR2019].
These eye fundus images are annotated with 8 labels indicating normal, 6 major ocular diseases and other ocular diseases.
The breakdown is shown in Table 1.

|              | train | test |     % |
|:------------ | ----: | ---: | ----: |
|Normal        |  2541 |  275 |  40.0 |
|Diabetes      |  1598 |  180 |  25.3 |
|Glaucoma      |   274 |   39 |   4.4 |
|Cataract      |   251 |   24 |   3.9 |
|AMD           |   250 |   30 |   4.0 |
|Hypertension  |   174 |   18 |   2.7 |
|Myopia        |   240 |   22 |   3.7 |
|Others        |   995 |  129 |  16.0 |
|Total         |  6323 |  717 | 100.0 |

**Table 1. Breakdown of eye fundus images for each disease label.**

## Methods

The progression of ocular disease can be represented by morphological lesions that are characteristic of each disease.
Since a patient can be affected by multiple diseases simultaneously, the detection and diagnosis of ocular disease is a multi-label classification model.
We employed *swapping assignments between views*, SwAV as a model[^SwAV2020].
The inference results of the model are obtained as ocular disease labels for the input fundus images.
We train the model on the training data and compute the AUC for each disease label and its average AUC for the predictions obtained for the test data.

## Results

For each disease, we obtained the highest AUC 1.00 for myopia and the lowest was 0.585 for hypertension, with a mean AUC of 0.886.

[^ODIR2019]: *Peking University International Competition on Ocular Disease Intelligent Recognition*, 2019.
[^SwAV2020]: *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*, **NeurIPS**, 2020.
