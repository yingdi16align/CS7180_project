import pandas as pd 
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

print('Loading the H1B dataset')

df = pd.read_csv("./1. Master H1B Dataset.csv",encoding='latin-1',low_memory=False)

# Keep only cases about h1b
df.drop(df.loc[df['VISA_CLASS']!='H1B'].index, inplace=True)

# Drop cases that applicant withdraw case
df.drop(df.loc[df['CASE_STATUS']=='CERTIFIEDWITHDRAWN'].index, inplace=True)
df.drop(df.loc[df['CASE_STATUS']=='WITHDRAWN'].index, inplace=True)

# calculate yearly wage for each case
df.loc[df['PW_UNIT_OF_PAY'] == 'Month', 'PREVAILING_WAGE'] = df['PREVAILING_WAGE'] * 12
df.loc[df['PW_UNIT_OF_PAY'] == 'Bi-Weekly', 'PREVAILING_WAGE'] = df['PREVAILING_WAGE'] * 26
df.loc[df['PW_UNIT_OF_PAY'] == 'Week', 'PREVAILING_WAGE'] = df['PREVAILING_WAGE'] * 52
df.loc[df['PW_UNIT_OF_PAY'] == 'Hour', 'PREVAILING_WAGE'] = df['PREVAILING_WAGE'] * 52 * 40
df.loc[df['WAGE_UNIT_OF_PAY'] == 'Month', 'WAGE_RATE_OF_PAY_FROM'] = df['WAGE_RATE_OF_PAY_FROM'] * 12
df.loc[df['WAGE_UNIT_OF_PAY'] == 'Bi-Weekly', 'WAGE_RATE_OF_PAY_FROM'] = df['WAGE_RATE_OF_PAY_FROM'] * 26
df.loc[df['WAGE_UNIT_OF_PAY'] == 'Week', 'WAGE_RATE_OF_PAY_FROM'] = df['WAGE_RATE_OF_PAY_FROM'] * 52
df.loc[df['WAGE_UNIT_OF_PAY'] == 'Hour', 'WAGE_RATE_OF_PAY_FROM'] = df['WAGE_RATE_OF_PAY_FROM'] * 52 * 40

# drop useless features
df = df.drop(['VISA_CLASS', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'EMPLOYER_COUNTRY', 'PW_UNIT_OF_PAY', 
              'PW_SOURCE_OTHER', 'WAGE_RATE_OF_PAY_TO', 'WAGE_UNIT_OF_PAY', 'WORKSITE_POSTAL_CODE'], axis=1)

# drop cases with missing value
df = df.dropna()

for column in df:
    if df[column].dtype == type(object):
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

y = df["CASE_STATUS"]
X = df.drop(["CASE_STATUS"], axis=1)

print('Training a Decision Tree classifier')
clf = DecisionTreeClassifier()

try:
  clf.fit(X, y)
except ValueError as e:
  print(e)

clf.fit(X, y)

print('Exporting the trained model')
joblib.dump(clf, 'model/h1b_classifier.joblib')