from dotenv import load_dotenv
import os

load_dotenv()  # lit automatiquement ton fichier .env

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    'architsharma01/loan-approval-prediction-dataset',
    path='./data',
    unzip=True
)

import pandas as pd
df = pd.read_csv('./data/loan_approval_dataset.csv')
print(df.shape)
print(df.head())
