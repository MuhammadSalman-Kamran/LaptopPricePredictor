#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('laptop_data.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# In[9]:


df.drop('Unnamed: 0', axis = 1, inplace= True)


# In[10]:


df.head()


# In[11]:


df['Ram'] = df['Ram'].str.replace('GB', '')
df['Ram'] = df['Ram'].astype('int32')


# In[12]:


df['Weight'] = df['Weight'].str.replace('kg','')
df['Weight'] = df['Weight'].astype('float32')


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


sns.distplot(np.log(df['Price']))


# In[16]:


df['Company'].value_counts().plot(kind = 'bar')


# In[17]:


sns.barplot(x = df['Company'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[18]:


df['TypeName'].value_counts().plot(kind = 'bar')


# In[19]:


sns.barplot(x = df['TypeName'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[20]:


sns.distplot(df['Inches'])


# In[21]:


sns.scatterplot(x = df['Inches'], y = df['Price'])


# In[22]:


df['ScreenResolution'].value_counts()


# In[23]:


df['TouchScreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)


# In[24]:


df['TouchScreen'].value_counts().plot(kind = 'bar')


# In[25]:


sns.barplot(x = df['TouchScreen'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[26]:


df.sample(5)


# In[27]:


df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)


# In[28]:


df['Ips'].value_counts().plot(kind = 'bar')


# In[29]:


sns.barplot(x = df['Ips'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[30]:


df.sample(5)


# In[31]:


df['Y_Res'] = df['ScreenResolution'].str.split('x', n= 1, expand = True)[1]


# In[34]:


new = df['ScreenResolution'].str.split('x', n=1 , expand = True)
df['X_Res'] = new[0].str.replace('\D', '')


# In[35]:


df.sample(10)


# In[36]:


df.info()


# In[37]:


df['X_Res'] = df['X_Res'].astype('int32')
df['Y_Res']= df['Y_Res'].astype('int32')


# In[38]:


df.corr()['Price']


# In[39]:


df['ppi'] = round(((df['X_Res']**2) + (df['Y_Res']**2)) ** 0.5/df['Inches'],2).astype('float32')


# In[40]:


df.head()


# In[41]:


df.drop('ScreenResolution', axis = 1, inplace = True)


# In[42]:


df.drop(['Inches', 'X_Res', 'Y_Res'], axis = 1, inplace = True)


# In[43]:


df.head()


# In[44]:


df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split(' ')[:3]))


# In[45]:


df.head()


# In[46]:


def processing(text):
    if text == 'Intel Core i5' or text == 'Intel Core i7' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD'


# In[47]:


df['Cpu Brand'] = df['Cpu Name'].apply(processing)


# In[48]:


df.sample(10)


# In[49]:


df['Cpu Brand'].value_counts().plot(kind='bar')


# In[50]:


sns.barplot(x = df['Cpu Brand'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[51]:


df.drop(['Cpu', 'Cpu Name'], axis = 1, inplace = True)


# In[52]:


df.head()


# In[53]:


df.Ram.value_counts().plot(kind = 'bar')


# In[54]:


sns.barplot(x = df['Ram'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[55]:


df['Memory'].value_counts()


# In[56]:


df['Memory']=df['Memory'].str.replace('\.0','')


# In[57]:


df['Memory'] = df['Memory'].str.replace('GB','')


# In[58]:


df['Memory'] = df['Memory'].str.replace('TB','000')


# In[59]:


new = df['Memory'].str.split('+', n = 1, expand= True)


# In[67]:


df['first'] = new[0]


# In[61]:


df['first'].str.strip()


# In[62]:


df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)


# In[63]:


df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)


# In[64]:


df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)


# In[65]:


df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)


# In[80]:


df['second'] = new[1]


# In[81]:


df['second'].fillna("0", inplace = True)


# In[82]:


df.sample(10)


# In[78]:


df['second'] = df['second'].fillna(0)


# In[83]:


df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in x else 0)


# In[84]:


df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in x else 0)


# In[85]:


df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)


# In[86]:


df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)


# In[87]:


df['first'] = df['first'].str.replace('\D', '')


# In[88]:


df['second'] = df['second'].str.replace('\D', '')


# In[89]:


df['first'] = df['first'].astype('int')
df['second'] = df['second'].astype('int')


# In[90]:


df['HDD'] = ((df['first'] * df['Layer1HDD']) + (df['second'] * df['Layer2HDD']))


# In[91]:


df['SSD'] = ((df['first'] * df['Layer1SSD']) + (df['second'] * df['Layer2SSD']))


# In[92]:


df['Hybrid'] = ((df['first'] * df['Layer1Hybrid']) + (df['second'] * df['Layer2Hybrid']))


# In[93]:


df['Flash_Storage'] = ((df['first'] * df['Layer1Flash_Storage']) + (df['second'] * df['Layer2Flash_Storage']))


# In[94]:


df.sample(10)


# In[95]:


df.columns


# In[96]:


df.drop(['first', 'second','Layer1HDD', 'Layer1SSD', 'Layer1Flash_Storage', 'Layer1Hybrid','Layer2HDD', 'Layer2SSD', 'Layer2Flash_Storage', 'Layer2Hybrid'], axis = 1, inplace=True)


# In[97]:


df.corr()['Price']


# In[98]:


df.drop('Memory', axis = 1,inplace = True)


# In[100]:


df.drop(['Hybrid','Flash_Storage'], axis =1 , inplace = True)


# In[102]:


df.sample(10)


# In[103]:


df['Gpu'].value_counts()


# In[106]:


df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0])


# In[107]:


df.head()


# In[115]:


df = df[df['Gpu Brand'] != 'ARM']


# In[117]:


sns.barplot(x = df['Gpu Brand'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[118]:


df.drop('Gpu', axis = 1, inplace = True)


# In[120]:


df['OpSys'].value_counts().plot(kind  ='bar')


# In[121]:


sns.barplot(x = df['OpSys'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[122]:


df['OpSys'].unique()


# In[124]:


def cat_op(text):
    if text == 'Windows 10' or text == 'Windows 10 S' or text == 'Windows 7':
        return 'Windows'
    elif text == 'Mac OS X' or text == 'macOS':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[126]:


df['OS'] = df['OpSys'].apply(cat_op)


# In[127]:


df.head()


# In[128]:


df.drop('OpSys', axis = 1, inplace = True)


# In[129]:


df['OS'].value_counts().plot(kind = 'bar')


# In[130]:


sns.barplot(x = df['OS'], y= df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[132]:


sns.displot(df['Weight'], kde = True)


# In[133]:


sns.scatterplot(x = df['Weight'], y = df['Price'])


# In[134]:


df.corr()['Price']


# In[136]:


plt.figure(figsize = (12, 8))
sns.heatmap(df.corr(), annot = True)


# In[137]:


df.head()


# In[173]:


df.to_csv('Clean_Laptop.csv')


# # Model Training

# In[164]:


pip install xgboost


# In[170]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# In[144]:


x = df.drop('Price', axis = 1)
y = np.log(df['Price'])


# In[145]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # Linear Regression

# In[147]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # Ridge Regression

# In[149]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = Ridge(alpha = 10)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # Lasso Regression

# In[150]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # KNN

# In[152]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # SVM

# In[153]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = SVR(kernel='rbf', C=10000, epsilon=0.1)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # Decision Tree

# In[155]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # AdaBoost

# In[159]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = AdaBoostRegressor(n_estimators=15, learning_rate=1.0)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # GradientBoosting

# In[168]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # ExtraTree

# In[161]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = ExtraTreesRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # RandomForest

# # XGBoost

# In[167]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = XGBRegressor(n_estimators=45,  max_depth=5, learning_rate = 0.5)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # RandomForest

# In[169]:


step1 = ColumnTransformer([
    ('cat_handle', OneHotEncoder(sparse=False, drop='first'),[0,1,7,10,11])
], remainder= 'passthrough')

step2 = RandomForestRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# In[172]:


pickle.dump(pipe, open('pipe.pkl','wb'))

