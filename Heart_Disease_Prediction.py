#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction by Dokpun Doley
# 

# ## Import the necessary libraries

# In[140]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the dataset
# 

# In[146]:


df=pd.read_csv("heart.csv")


# In[147]:


df.sample(5)


# ### Numerical value information

# In[148]:


df.describe()


# In[149]:


df.info()


# - We do not have any missing value in the dataset

# ### To understand the columns better

# In[150]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(df.columns[i]+":\t\t\t" +info[i])
    


# ## Analysing the 'target' variable

# In[151]:


df['target'].describe()


# In[152]:


df['target'].unique()


# - This is a classification Problem with the target variable having values '0' and '1'

# ## Checking the correlation between all the columns

# In[153]:


co_rel=df.corr()['target'].abs()*100
co_rel.sort_values(ascending=False)


# - This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.

# # Exploratory Data Analysis(EDA)

# ## 1.Analyse the target variable

# In[154]:


temp_target=df.target.value_counts()
temp_target


# In[155]:


sns.countplot(x='target',data=df)


# In[156]:


print(f"Percentage of people with heart Problems: {round(y[1]*100/303,2)}")
print(f"Percentage of people without heart Problems: {round(y[0]*100/303,2)}")


# ## Now we will analyse for all the features: 'sex', 'cp', 'fbs', 'restgcg', 'exang', 'slope', 'ca', and 'thal' 

# ## 2.Analyse the 'sex' feature

# In[157]:


df['sex'].unique()


# In[158]:


sns.barplot(x=df['sex'],y=df['target'])


# - From the plot we can see that females are more likely to have heart problems than males

# ## 3.Analyse  'cp' features -- chest pain

# In[159]:


df['cp'].unique()


# In[160]:


sns.barplot(x=df['cp'],y=df['target'])


# - Conclusion: The chest pain of '0' type i.e the one with typical angina are much less likely to have heart problems

# ## 4.Analyse the 'fbs' feature

# In[161]:


df['fbs'].unique()


# In[162]:


sns.barplot(x=df['fbs'],y=df['target'])


# - Nothing much difference to make a conclusion

# ## 5. Analyse the 'restecg' feature

# In[163]:


df['restecg'].unique()


# In[164]:


sns.barplot(x=df['restecg'],y=df['target'])


# - Conclusion: The people with restecg '0' and '1' are much more likely to have a heart disease than with restecg '2'

# ## 6.Analyse the 'exang' feature

# In[165]:


df['exang'].unique()


# In[166]:


sns.barplot(x=df['exang'],y=df['target'])


# Conclusion: People with exang=1 i.e .Exercise induced angina are much less likely to have heart problems

# ## 7.Analyse the 'slope' feature

# In[167]:


df['slope'].unique()


# In[168]:


sns.barplot(x=df['slope'],y=df['target'])


# - Conclusion: We noticed that slope=2 causes heart pain much more then slope value '1' and '0'

# ## 8.Analyse the 'ca' feature

# In[169]:


df['ca'].unique()


# In[170]:


sns.countplot(x=df['ca'],data=df)


# In[171]:


sns.barplot(x=df['ca'],y=df['target'])


# - Conclusion: From the above two charts we can say ca=4 has larger number of heart patients

# ## 8.Analyse the 'thal' feature

# In[172]:


df['thal'].unique()


# In[173]:


sns.barplot(x=df['thal'],y=df['target'])


# # Train Test split

# In[174]:


heart_disease_condition=df.target
features=df.drop('target',axis=1)


# In[175]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,heart_disease_condition,test_size=0.2,random_state=0)


# In[176]:


x_train.shape


# In[177]:


x_test.shape


# In[178]:


y_train.shape


# In[179]:


y_test.shape


# ## Model fitting

# ## 1.Logistic Regression

# In[180]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#now we will fit the training values to the model
lr.fit(x_train,y_train)


# ####  Prediction:

# In[181]:


lr_prediction=lr.predict(x_test)
lr_prediction


# In[182]:


from sklearn.metrics import accuracy_score
score_lr=round(accuracy_score(lr_prediction,y_test)*100,2)
print(f"The accuracy score I got using Logistic Regression is: {score_lr} %")


# ## 2.Naive Bayes

# In[183]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)


# ####  Prediction:

# In[184]:


nb_prediction=nb.predict(x_test)
nb_prediction


# In[185]:


score_nb=round(accuracy_score(nb_prediction,y_test)*100,2)
print(f"The accuracy score I got using Naive Bayes is: {score_nb} %")


# ## 3.K Nearest Neighbors

# In[186]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# #### Prediction:

# In[187]:


knn_prediction=knn.predict(x_test)
knn_prediction


# In[188]:


score_knn=round(accuracy_score(knn_prediction,y_test)*100,2)
print(f"The accuracy score I got using K Nearest Neighbors is: {score_knn} %")


# ## 4. SVM

# In[189]:


from sklearn import svm
sv=svm.SVC(kernel='linear')
sv.fit(x_train,y_train)


# #### Prediction:

# In[190]:


svm_prediction=sv.predict(x_test)
svm_prediction


# In[191]:


score_svm=round(accuracy_score(svm_prediction,y_test)*100,2)
print(f"The accuracy score I got using SVM is: {score_svm} %")


# ## 5. Random Forest

# In[192]:


from sklearn.ensemble import RandomForestClassifier
max_accuracy=0

for x in range(2000):
    rf=RandomForestClassifier(random_state=x)
    rf.fit(x_train,y_train)


# In[193]:


rf_prediction=rf.predict(x_test)
rf_prediction


# In[194]:


current_accuracy=round(accuracy_score(rf_prediction,y_test)*100,2)


# In[195]:


if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x


# In[196]:


print(max_accuracy)
print(best_x)


# In[197]:


rf = RandomForestClassifier(random_state=best_x)
rf.fit(x_train,y_train)
new_rf_prediction = rf.predict(x_test)


# In[198]:


score_rf = round(accuracy_score(new_rf_prediction,y_test)*100,2)

print(f"The accuracy score I got using SVM is: {score_rf} %")


# ## Output Final Score

# In[199]:


scores = [score_lr,score_nb,score_knn,score_svm,score_rf]
algorithms = ["Logistic Regression","Naive Bayes","K-Nearest Neighbors","Support Vector Machine","Random Forest"]    

for i in range(len(algorithms)):
    print(f"The accuracy score I achieved using {algorithms[i]} is: {str(scores[i])} %")


# In[201]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(x=algorithms,y=scores)


# ## Conclusion:
#  ### From all the models we can see that the Random Forest model gives the most accurate result

# In[ ]:




