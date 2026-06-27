# Customer Segmentation using K-Means Clustering

## Overview

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to segment customers based on demographic and socio-economic characteristics. The workflow includes data preprocessing, feature scaling, cluster optimization, model training, and visualization of customer segments.

## Dataset

The dataset contains the following features:

* ID
* Sex
* Marital Status
* Age
* Education
* Income
* Occupation
* Settlement Size

The `ID` column is excluded from the clustering process as it does not contribute to customer segmentation.

## Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure


Customer-Segmentation-KMeans/
│
├── data/
├── notebooks/
├── results/
├── images/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


## Methodology

* Data preprocessing
* Feature scaling using StandardScaler
* Optimal cluster selection using the Elbow Method and Silhouette Score
* Customer segmentation with K-Means Clustering
* Cluster visualization using Principal Component Analysis (PCA)
* Cluster profiling and result export

## Results

The model groups customers into distinct clusters based on similar characteristics, providing valuable insights for customer profiling and targeted marketing strategies.


## Author

Olumide Adeyanju

