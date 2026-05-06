# Decoding Risk Factors of ALZ: how to classify Alzheimer’s disease using demographic, health, lifestyle, and cognitive variables. 
# The code explains preprocessing, exploratory analysis, dimensionality reduction (LLE) and final model evaluation (DT/SVM).

## Opening the Project #####
#The project can be run in Python or in Visual Studio Code.
#Place the following files in the same folder: 
alzheimers_disease_data.csv; preprocessing.ipynb
#Run code:

## Installing Packages #####

Install the required libraries before running the program: 
pip install pandas numpy matplotlib seaborn plotly scikit-learn

## Loading the Dataset ####
#Update the folder path so it points to where your dataset is stored and load your dataset:

os.chdir("/your/project/folder")
df_raw = pd.read_csv("alzheimers_disease_data.csv")

## What the Code is Doing #####
#Cleaning and Preprocessing: Identifier columns  removed; Ethnicity variable is converted into one-hot encoded columns.
#Two datasets are then created:
## full dataset containing all variables
## risk-only dataset where symptom variables are removed

#Code then performs exploratory analysis, including:
##descriptive statistics
##histograms
##distribution plots by diagnosis group
##outlier detection
##correlation 

#After preprocessing, LLE reduces dimensionality of dataset.
#Then following models are compared:
##Decision Tree (DT)
##Linear kernel Support Vector Machine (SVM)
##RBF kernel Support Vector Machine (SVM)
###Each model is evaluated: with LLE, without LLE, and on both the full and risk-only datasets

## Evaluation
#Models are evaluated using:ROC-AUC, Accuracy, Precision, Recall, F1-score, Confusion matrices, ROC curves
#Decision Tree models have feature importance plots and tree visualizations

## ALSO ##
#Train/test split: 70/30, 
#random_state=42: used for reproducibility
#stratify=y: class proportions same across train/test sets  
