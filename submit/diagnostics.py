
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(dataset):
    #read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path + '/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    testdata=pd.read_csv(test_data_path+'/'+dataset)
    X_test = testdata.loc[: ,['lastmonth_activity','lastyear_activity','number_of_employees']]
       
    predicted = model.predict(X_test)
    
    return predicted #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    final_dataset=pd.read_csv(dataset_csv_path+'/finaldata.csv')
    X = final_dataset.loc[: ,['lastmonth_activity','lastyear_activity','number_of_employees']]
    summary_statistics = []
    for column in X:
        summary_statistics.extend([X[column].mean(), X[column].median(), X[column].std()])
        
    return summary_statistics#return value should be a list containing all summary statistics

##################Function to get summary statistics
def dataframe_missing_data():
    #calculate percent of missing data here
    final_dataset=pd.read_csv(dataset_csv_path+'/finaldata.csv')
    percent_nas=list((final_dataset.isna().sum())/len(final_dataset.index))
   
    return percent_nas

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime1 = timeit.default_timer()
    os.system('python3 training.py')
    timing_training=timeit.default_timer() - starttime1
    
    starttime2 = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ingestion=timeit.default_timer() - starttime2
    
    return [timing_training, timing_ingestion] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of intalled dependecies and theis version and the most recent version
    intalled_modules = subprocess.check_output(['pip', 'list'])
    with open('intalled_modules.txt', 'wb') as f:
           f.write(intalled_modules)
    intalled_modules_df = pd.read_fwf('intalled_modules.txt')
    intalled_modules_df = intalled_modules_df.iloc[1: , :].reset_index(drop=True)
    
    outdated_modules = subprocess.check_output(['pip', 'list','--outdated'])
    with open('outdated_modules.txt', 'wb') as f:
           f.write(outdated_modules)      
    outdated_modules_df = pd.read_fwf('outdated_modules.txt')
    outdated_modules_df = outdated_modules_df.iloc[1: , :].set_index('Package')
    
    intalled_modules_df['Latest'] = intalled_modules_df['Version']

    for index, row in outdated_modules_df.iterrows():
        intalled_modules_df['Latest'].loc[intalled_modules_df['Package'] == index] = row[1]
    
    return intalled_modules_df
    
if __name__ == '__main__':
    model_predictions('testdata.csv')
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
