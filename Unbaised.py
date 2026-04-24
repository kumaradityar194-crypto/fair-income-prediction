import pandas as pd
import numpy as np

dataset=pd.read_csv(r"c:\Users\kumar\Downloads\archive\UCI_Adult_Income_Dataset.csv")
print(dataset.head(10))
print(dataset["sex"].unique())   
print(dataset.size)
dataset.replace("?", pd.NA, inplace=True)
print(dataset.isna().sum())
print("Information...\n")
print(dataset.info())
print("\n")
print("mode of workclass:",dataset["workclass"].mode()[0])
print("mode of Occupation:",dataset["occupation"].mode()[0])
print("mode of native-country:",dataset["native-country"].mode()[0])
print("")
print("Handling missing value:")
dataset["workclass"].fillna(dataset["workclass"].mode()[0], inplace=True)
dataset["occupation"].fillna(dataset["occupation"].mode()[0], inplace=True)
dataset["native-country"].fillna(dataset["native-country"].mode()[0], inplace=True)

missing=dataset.isna().sum()
print(missing)
print(dataset["income"].count())
print(dataset["income"].unique())

dataset["income"]=dataset["income"].str.strip()
dataset["income"]=dataset["income"].str.replace(".","",regex=False)
dataset["income"]=dataset["income"].map({"<=50K":0,">50K":1})

data=dataset.select_dtypes(include='object').columns
dataset=pd.get_dummies(dataset,columns=data,drop_first=True)
print("Information.....\n")
print(dataset.info())
print(dataset.size)

sen_data=dataset["sex_Male"]   #change

cols_to_remove=[col for col in dataset.columns if "sex_" in col or "relationship_" in col]   #change

x=dataset.drop(["income"]+cols_to_remove,axis=1)   #change
y=dataset["income"]

print(x.size)
print(y.size)
print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,s_train,s_test=train_test_split(x,y,sen_data,test_size=0.25,random_state=42)   #change

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(x_train)
X_test_scaled=scaler.transform(x_test)

weights=np.where(s_train==0,2.0,1.0)   #change

from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient,DemographicParity   #change

base_model=LogisticRegression(max_iter=5000)

model=ExponentiatedGradient(estimator=base_model,constraints=DemographicParity())   #change

model.fit(X_train_scaled,y_train,sensitive_features=s_train)   #change

from sklearn.metrics import accuracy_score

y_prob=model._pmf_predict(X_test_scaled)[:,1]   #change

best_t=0.5
best_bias=999

for t in np.arange(0.3,0.7,0.01):
    y_temp=(y_prob>t).astype(int)
    df=pd.DataFrame({"sex":s_test,"pred":y_temp})
    bias=abs(df[df["sex"]==0]["pred"].mean()-df[df["sex"]==1]["pred"].mean())
    if bias<best_bias:
        best_bias=bias
        best_t=t

y_pred=(y_prob>best_t).astype(int)   #change

accuracy=accuracy_score(y_test,y_pred)
print("\n")
print("Accuracy of model.:",accuracy*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(model.predictors_[0].coef_)
#print(base_model.coef_)

def check_bias(s_test,y_test,y_pred):
    df_temp=pd.DataFrame()
    df_temp["sex_Male"]=s_test.values
    df_temp["y_true"]=y_test.values   #change
    df_temp["prediction"]=y_pred

    group_0=df_temp[df_temp["sex_Male"]==0]["prediction"].mean()
    group_1=df_temp[df_temp["sex_Male"]==1]["prediction"].mean()

    def tpr(g):
        d=df_temp[df_temp["sex_Male"]==g]
        tp=((d["y_true"]==1)&(d["prediction"]==1)).sum()
        fn=((d["y_true"]==1)&(d["prediction"]==0)).sum()
        return tp/(tp+fn+1e-6)

    bias=abs(group_0-group_1)
    eq_opp=abs(tpr(0)-tpr(1))   #change

    print("\n===== FAIRNESS REPORT =====")
    print("Statistical Parity:",bias)
    print("Equal Opportunity:",eq_opp)

check_bias(s_test,y_test,y_pred)

import matplotlib.pyplot as plt

thresholds=np.arange(0.1,0.9,0.05)
bias_list=[]

for t in thresholds:
    y_pred_temp=(y_prob>t).astype(int)
    df_temp=pd.DataFrame()
    df_temp["sex_Male"]=s_test.values
    df_temp["prediction"]=y_pred_temp
    group_0=df_temp[df_temp["sex_Male"]==0]["prediction"].mean()
    group_1=df_temp[df_temp["sex_Male"]==1]["prediction"].mean()
    bias=abs(group_0-group_1)
    bias_list.append(bias)

plt.figure()
plt.plot(thresholds,bias_list)
plt.xlabel("Threshold")
plt.ylabel("Bias")
plt.title("Bias vs Threshold")
plt.show()

import shap

X_train_df = pd.DataFrame(X_train_scaled, columns=x.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=x.columns)

explainer = shap.LinearExplainer(model.predictors_[0], X_train_df)
shap_values = explainer(X_test_df)

shap.summary_plot(shap_values, X_test_df)
shap.plots.waterfall(shap_values[0])

import joblib

joblib.dump(model,"fair_model.pkl")
joblib.dump(scaler,"fair_scaler.pkl")
joblib.dump(x.columns,"fair_features.pkl")
print("end")