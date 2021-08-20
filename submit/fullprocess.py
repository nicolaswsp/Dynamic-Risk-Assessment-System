import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import ast
import subprocess
import pickle
import json
import os

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Check and read new data
#first, read ingestedfiles.txt
with open(prod_deployment_path+'/ingestedfiles.txt', 'r') as f:
    ingested_files = ast.literal_eval(f.read())
    
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_path = os.getcwd()+'/'+input_folder_path+'/'
filenames = os.listdir(input_path)

new_data = False
for each_filename in filenames:
    if each_filename not in ingested_files:
        print('The new file {} was found'.format(each_filename))
        new_data = True
        
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_data:
    exec(open("./ingestion.py").read())
else:
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(prod_deployment_path+'/latestscore.txt', 'r') as f:
    latest_score = float(f.read())

model_prediction = scoring.score_model(prod_deployment_path, 'trainedmodel.pkl', dataset_csv_path, 'finaldata.csv')


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if model_prediction < latest_score:
    print('Model drift has occurred')
else:
    exit()

##################Re-training
#if you found evidence for model drift, re-run the training.py script
exec(open("./training.py").read())
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
exec(open("./deployment.py").read()) 
##################Diagnostics and reporting
#run appicalls.py and reporting.py for the re-deployed model
exec(open("./reporting.py").read()) 
exec(open("./apicalls.py").read()) 









