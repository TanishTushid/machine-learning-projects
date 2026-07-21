

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', None)

df = pd.read_csv(r"C:\Users\TanishTushid\OneDrive\Desktop\Employee-Attrition-Prediction\data\WA_Fn-UseC_-HR-Employee-Attrition (1).csv")

print(df.head())
print(df.columns)
print(df.describe())
print(df.info())


df.drop(columns = ["EmployeeCount",
                  "EmployeeNumber",
                   "Over18",
                   "StandardHours"
                  ], inplace=True)
                

print(df.shape)


categorical_cols = df.select_dtypes(include=['object', 'str']).columns
numerical_cols = df.select_dtypes(include=['number']).columns

print(categorical_cols)
print(numerical_cols)

# encoding categorical data

print(df[categorical_cols].head())
print(df[categorical_cols].describe())

# label encode categorical data 

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Attrition'] = le.fit_transform(df['Attrition'])
df['OverTime'] = le.fit_transform(df['OverTime'])

print(df[categorical_cols].head())

# one hot encoder for remaining 
ohe_cols = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'JobRole',
    'MaritalStatus'
]

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

encoded_df = ohe.fit_transform(df[ohe_cols])

df = df.drop(columns=ohe_cols)
df = pd.concat([df, encoded_df], axis=1)

# visualization

plt.figure(figsize=(5, 4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.show()

df[numerical_cols].hist(figsize=(18,15), bins=20)
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    annot=False,
)

plt.title("Correlation Heatmap")
plt.show()


x= df.drop(columns='Attrition')
y = df['Attrition']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)

rf = RandomForestClassifier(n_estimators=1000, criterion="entropy",
                            max_depth=14,
                            min_samples_split=10,
                            random_state=42
    )
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print(f"prediction : {y_pred}")


print(f"confusion matrix : {classification_report(y_test, y_pred)} ")

importance = pd.Series(
    rf.feature_importances_,
    index=x.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(
    x=importance[:10],
    y=importance.index[:10]
)

plt.title("top 10 important features")
plt.show()

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = SVC()
model.fit(x_train_scaled, y_train)

y2_pred = model.predict(x_test_scaled)
print(f"prediction {y2_pred}")

print(f"confusion matrix: {confusion_matrix(y_test, y2_pred)}")

print(f"classification report : {classification_report(y_test, y2_pred)}")

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb.fit(x_train,y_train)
y3_pred=xgb.predict(x_test)
print("Predictions:")
print(y3_pred)

print(f"confusion matrix: {confusion_matrix(y_test, y3_pred)}")

print(f"classification report : {classification_report(y_test, y3_pred)}")


lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(x_train_scaled, y_train)

y4_pred = lr.predict(x_test_scaled)

print("Predictions:")
print(y4_pred)

print(f"confusion matrix: {confusion_matrix(y_test, y4_pred)}")

print(f"classification report : {classification_report(y_test, y4_pred)}")




nb = GaussianNB()

nb.fit(x_train, y_train)

y5_pred = nb.predict(x_test)

print("Predictions:")
print(y5_pred)

print(f"confusion matrix: {confusion_matrix(y_test, y5_pred)}")

print(f"classification report : {classification_report(y_test, y5_pred)}")



knn = KNeighborsClassifier(
    n_neighbors=5
)
knn.fit(x_train_scaled, y_train)

y6_pred = knn.predict(x_test_scaled)

print("Predictions:")
print(y6_pred)



print(f"confusion matrix: {confusion_matrix(y_test, y6_pred)}")

print(f"classification report : {classification_report(y_test, y6_pred)}")

rf_acc = accuracy_score(y_test, y_pred)
svm_acc = accuracy_score(y_test, y2_pred)
xgb_acc = accuracy_score(y_test, y3_pred)
lr_acc = accuracy_score(y_test, y4_pred)
nb_acc = accuracy_score(y_test, y5_pred)
knn_acc = accuracy_score(y_test, y6_pred)

results = pd.DataFrame({

    "Model":[
        "Random Forest",
        "SVM",
        "XGBoost",
        "Logistic Regression",
        "Naive Bayes",
        "KNN"
    ],

    "Accuracy":[
        rf_acc,
        svm_acc,
        xgb_acc,
        lr_acc,
        nb_acc,
        knn_acc
    ]
})

print(results.sort_values(by="Accuracy", ascending=False))
plt.figure(figsize=(8,5))

sns.barplot(
    data=results,
    x='Accuracy',
    y='Model'
)

plt.title("Model Accuracy Comparison")
plt.show()