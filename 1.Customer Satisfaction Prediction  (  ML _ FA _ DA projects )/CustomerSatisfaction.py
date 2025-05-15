import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('customer_support_tickets.csv')
df = data.dropna()
df = df.drop_duplicates()

label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    # Fit the encoder on ALL unique values in the column
    label_encoders[column].fit(df[column])
    df.loc[:, column] = label_encoders[column].transform(df[column]) # Use transform after fitting

# Save the label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)
print('Label encoders have been pickled and saved as label_encoders.pkl')

X = df.drop(['Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase', 'Ticket Description', 'Resolution', 'First Response Time', 'Time to Resolution', 'Customer Satisfaction Rating'], axis=1)
y = df['Customer Satisfaction Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

file_name = 'customer_satisfaction.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(rfc, file)

print('Model has been pickled and saved as customer_satisfaction.pkl')