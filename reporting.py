import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model(test_dataset):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    testdata=pd.read_csv(test_data_path+'/testdata.csv')
    y_test = testdata['exited']
   
    predicted = model_predictions(test_dataset)
    cf_matrix = metrics.confusion_matrix(y_test, predicted)

    # Create a confusion matrix using seaborn heatmap
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.savefig(output_model_path+'/confusionmatrix2.png')
    
    
if __name__ == '__main__':
    score_model('testdata.csv')
