#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact


# In[3]:


df = pd.read_csv("Crop.csv")
df.head()
# df.shape


# ### Check if there is any missing value

# In[4]:


df.isnull().sum()


# In[5]:


import missingno as msno
msno.matrix(df)


# In[6]:





# In[7]:


df['label'].value_counts()


# In[8]:


mean = pd.Series.mean(df)
mean


# ### Checking values for individual crop

# In[9]:


@interact
def summary(crops = list(df['label'].value_counts().index)):
    x = df[df['label']==crops]
    print("###################################")
    print("Nitrogen")
    print("Min: ",x['N'].min())
    print("Avg: ",x['N'].mean())
    print("Max: ",x['N'].max())
    print("###################################")
    print("Phosphorous")
    print("Min: ",x['P'].min())
    print("Avg: ",x['P'].mean())
    print("Max: ",x['P'].max())
    print("###################################")
    print("Potassium")
    print("Min: ",x['K'].min())
    print("Avg: ",x['K'].mean())
    print("Max: ",x['K'].max())
    print("###################################")
    print("Temperature")
    print("Min: ",x['temperature'].min())
    print("Avg: ",x['temperature'].mean())
    print("Max: ",x['temperature'].max())
    print("###################################")
    print("Humidity")
    print("Min: ",x['humidity'].min())
    print("Avg: ",x['humidity'].mean())
    print("Max: ",x['humidity'].max())
    print("###################################")
    print("PH")
    print("Min: ",x['ph'].min())
    print("Avg: ",x['ph'].mean())
    print("Max: ",x['ph'].max())
    print("###################################")
    print("Rainfall")
    print("Min: ",x['rainfall'].min())
    print("Avg: ",x['rainfall'].mean())
    print("Max: ",x['rainfall'].max())


# ### Checking values for individual weather

# In[10]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    list = df.label.values.tolist()
    list = np.unique(list)
    
    lst = ['rice','mango']
    for i in list:
        print(i.capitalize()+ " : "+ format(df[(df['label']==i)][conditions].mean()))


# ### Check the values higher and lower than average

# In[11]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Greater than Avg")
    print(df[df[conditions] > df[conditions].mean()]['label'].unique())
    print("\n")
    print("Less than Avg")
    print(df[df[conditions] <= df[conditions].mean()]['label'].unique())


# ### Graphs for each weather condition

# In[12]:


plt.subplot(2,4,1)
sns.distplot(df['N'],color="black")
plt.xlabel("Ratio of Nitrogen",fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(df['P'],color="black")
plt.xlabel("Ratio of Phosphorous",fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(df['K'],color="black")
plt.xlabel("Ratio of Potassium",fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(df['temperature'],color="black")
plt.xlabel("Ratio of Potassium",fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(df['rainfall'],color="black")
plt.xlabel("Ratio of Rainfall",fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(df['humidity'],color="black")
plt.xlabel("Ratio of Humidity",fontsize = 12)
plt.grid()


# ### Seasonal Crops

# In[13]:


print("Summer Crops")
print(df[(df['temperature']>30) & (df['humidity']>50)]['label'].unique())
print("\n")
print("Winter Crops")
print(df[(df['temperature']<20) & (df['humidity']>30)]['label'].unique())
print("\n")
print("Rainy Crops")
print(df[(df['rainfall']>200) & (df['humidity']>30)]['label'].unique())
print("\n")


# ### Applying Kmeans clustering

# In[14]:


from sklearn.cluster import KMeans
# removing the label column
x = df.drop(['label'],axis=1)

# selecting all the values of the data
x=x.values

# checking the shape
print(x.shape)


# In[15]:


plt.rcParams['figure.figsize'] = (10,4)
    
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

# plotting the results
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of clusters')
plt.ylabel('wcss')
plt.show()


# In[16]:


km = KMeans(n_clusters = 4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means = km.fit_predict(x)

#finding the results
a = df['label']
y_means = pd.DataFrame(y_means)
y_means
z = pd.concat([y_means,a],axis =1)
z = z.rename(columns = {0:'cluster'})
z

#check each cluster
print("1st Cluster: ",z[z['cluster']==0]['label'].unique())
print("2nd Cluster: ",z[z['cluster']==1]['label'].unique())
print("3rd Cluster: ",z[z['cluster']==2]['label'].unique())
print("4th Cluster: ",z[z['cluster']==3]['label'].unique())


# ### Split dataset for training model

# In[17]:


y = df['label']
x = df.drop(['label'],axis = 1)
print(x.shape)
print(y.shape)


# In[18]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[19]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[20]:


from sklearn.metrics import confusion_matrix

plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap = 'Wistia')
plt.title('Confusion matrix for logistic regression',fontsize = 15)
plt.show()


# In[21]:


from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)


# In[22]:


df.head()


# In[23]:


pred = model.predict(np.array(np.array([[90,40,40,20,80,7,50]])))
print(pred)


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# In[25]:


x=df.drop(["label"],axis=1)
y=df['label']


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['label']),
                                                 df['label'],
                                                 test_size=0.2,
                                                random_state=42)


# In[27]:


x_train


# In[28]:


tr=ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])


# In[29]:


tr


# In[30]:


tr2=SelectKBest(score_func=chi2,k=7)
tr2


# In[31]:


from sklearn.linear_model import LogisticRegression
tr3=LogisticRegression()


# In[32]:


pipe=make_pipeline(tr,tr2,tr3)


# In[33]:


pipe.fit(x_train,y_train)


# In[34]:


y_pred=pipe.predict(x_test)


# In[35]:


y_pred
y_pred.shape


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[37]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




