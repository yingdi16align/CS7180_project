import pandas as pd 
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

print('Loading the H1B dataset')

df = pd.read_csv("H1B Disclosure Dataset Files/1. Master H1B Dataset.csv",encoding='latin-1',low_memory=False)

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
df = df.drop(['CASE_SUBMITTED_DAY', 'CASE_SUBMITTED_MONTH', 'CASE_SUBMITTED_YEAR', 'VISA_CLASS', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'EMPLOYER_COUNTRY', 'PW_UNIT_OF_PAY', 
              'PW_SOURCE_OTHER', 'WAGE_RATE_OF_PAY_TO', 'WAGE_UNIT_OF_PAY', 'WORKSITE_POSTAL_CODE'], axis=1)

# Remove the space
df.loc[df['SOC_NAME'] == 'ENGINEERS ', 'SOC_NAME'] = 'ENGINEERS'

# drop cases with missing value
df = df.dropna()

for column in df:
    if df[column].dtype == type(object):
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

y = df["CASE_STATUS"]
X = df.drop(["CASE_STATUS"], axis=1)

seed = 7
test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Decision Tree classifier
print('Training a Decision Tree classifier')
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

score = clf_dt.score(X_test, y_test)
print('The test score for Decision Tree: ' + str(score))

print('Exporting the trained model')
joblib.dump(clf_dt, 'model/h1b_classifier_dt.joblib')

# Random Forest classifier
print('Training a Random Forest classifier')
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

score = clf_rf.score(X_test, y_test)
print('The test score for Random Forest: ' + str(score))

print('Exporting the Random Forest trained model')
joblib.dump(clf_rf, 'model/h1b_classifier_rf.joblib')

# Naive Bayes classifier
print('Training a Naive Bayes classifier')
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)

score = clf_nb.score(X_test, y_test)
print('The test score for Naive Bayes: ' + str(score))

print('Exporting the Naive Bayes trained model')
joblib.dump(clf_nb, 'model/h1b_classifier_nb.joblib')