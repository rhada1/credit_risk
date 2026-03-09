import pandas as pd

df = pd.read_csv('./data/loan_approval_dataset.csv')
df.columns = df.columns.str.strip()
df['loan_status'] = df['loan_status'].str.strip()

print(df.groupby('loan_status')['cibil_score'].describe())
print("\nApproved - cibil min:", df[df['loan_status']=='Approved']['cibil_score'].min())
print("Rejected - cibil max:", df[df['loan_status']=='Rejected']['cibil_score'].max())