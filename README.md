# Comparing-feature-extraction-methods-with-BioTrove-dataset
## Overview
This repo covers a research project that compared different deep learning/machine learning methods for feature extraction in images. This project took place in the greater context of the ["clustering-biotrove"](https://www.kaggle.com/competitions/biotrove-clustering/overview) challenge on Kaggle. This challenge involved a subset of the [BioTrove](https://baskargroup.github.io/BioTrove/) biodiversity dataset which in original form contains over 160 million images of living organisms with taxonomic information included. The subset simply contained 50k images and **only taxonomic family labels**. The challenge is to use unsupervised learning (no genus or species labels provided) to produce genus- and species-level clusters. See the sections on the original BioTrove dataset and the Clustering BioTrove Challenge section for more information on each respective topic. The general appraoch of my team, BioTrove_1, was to use a deep learning approach for feature/embedding extraction from the images which would then allow for clustering of the images into genus- and species-level groupings. My research question for this project was to compare **four** different feature extraction methods from models that I designed to determine which approach resulted in better-defined embeddings when visualized using UMAP for dimension-reductionality.

## Note on useage of AI 
LLMs including ChatGPT, Gemini, and the Acanaconda Navigator, which uses several open-source LLM models, were used throughout this project to debug code and structure certain sections - particularly the custom dataset class, the supervised contrastive learning loss function, and the supervised contrastive learning model. 

## Scripting language & library
Code is in **Python** and the deep learning package used for analysis is **PyTorch**. 
Code is presented in the form of Jupyter notebooks.

## Repo structure - How to use this repo
To keep things clear and organized, this repo is broken up into **different sections in the main branch**. 
The README file includes the following:
  - Information on the original biotrove dataset
  - Information on the clustering biotrove challenge and the BioTrove data subset used for this challenge
  - A description of the research proejct/experiment which is the subject matter of this repository with visualizations of embeddings

The **code to reproduce** the results of **each of the four feature extraction methods** is **seperated into different files** in the main branch:
  - The **Data Preprocessing** file contains code to load in the clustering biotrove image dataset and metadata (49,633 image subset of original BioTrove dataset with corresponding metadata csv file), perform    transforms on images, custom dataset subclass, and dataloader. Since all four feature extraction methods function using the same preprocessing strategy, a seperate file has been used for this code. 
  - Each of the four extraction methods has its own, uniquely named section. Each individual section includes an explanation of the particular extraction method used as well as code to perform that method. The    four extraction method files include:
      - **Single-layer ResNet50 feature extraction**
      - **Double-layer ResNet50 feature extraction**
      - **Double-layer ResNet101 feature extraction**
      - **Double-layer ResNet50 + supervised contrastive learning feature extraction**
  - The **Embedding Visualization** file contains information on the visualization method used as wel as the code to visualize the embeddings extracted using any of the extraction methods listed above. Since only one visualization method was used, and it is compatible with all the extraction methods, the visualizaton method exists as an individual file.

My recommendation for **reproducing the experiment** is to open up a blank notebook (jupyter, colab, etc.), and **copy and paste the code** from the "Data Preprocessing" file, the desired feature extraction method file, and "Embedding Visualization" file into the notebook. Each individual file includes the import statements necessary for that specific code at the top and is designed to be ready to use as is. The **only edit necessary** is the addition of your **unique file paths** to the image directory and the metadata csv. **Instructions for downloading the data** are included in the Data Preprocessing file.

## Background
### Original BioTrove Dataset
The original BioTrove dataset...

### Clustering BioTrove Challenge
The "Clustering BioTrove Challenge" was...

## Research Question: Which feature extraction method performs the best (i.e. creates the most separability for embeddings in embedding space)?


