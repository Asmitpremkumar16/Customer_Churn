import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data= pd.read_csv("churn modelling.csv")

X= data.drop(columns= ["RowNumber","CustomerId","Surname", "Exited"])

numeric_cols= ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
categorical_cols = ['Geography', 'Gender']
passthrough_cols = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']

preprocessor= ColumnTransformer(transformers= [
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop= 'first', sparse_output= False), categorical_cols),
    ('pass', 'passthrough', passthrough_cols)
    ])

preprocessor.fit(X)
joblib.dump(preprocessor, 'preprocessor.pkl')
print("preprocessor.pkl saved successfully!")