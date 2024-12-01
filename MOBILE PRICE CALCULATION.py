#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\python\mobile price project 2\train.csv")
df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.sample()


# In[11]:


df.shape


# In[12]:


df['battery_power'].isnull().sum()


# In[13]:


df.clock_speed.mean()


# In[14]:


df.int_memory.median()


# In[15]:


df.mobile_wt.mode()


# In[16]:


df.sc_h.var()


# In[17]:


df.price_range.std()


# In[18]:


plt.bar(height=df.px_width,x=np.arange(1,2001,1))


# In[19]:


plt.hist(df.px_height)
plt.hist(df.px_height,bins=[143,165,245,379,650,711,891,923,1045],color='yellow',edgecolor='pink')


# In[20]:


plt.hist(df.px_width)
plt.hist(df.px_width,bins=[123,357,489,724,895,945,1067],color='purple',edgecolor='m')
help(plt.hist)


# In[21]:


import seaborn as sns


# In[22]:


sns.distplot(df.px_height)
sns.displot(df.px_width)


# In[23]:


plt.figure()
plt.boxplot(df.px_height)
help(plt.boxplot)


# In[24]:


df.loc[1:11,['mobile_wt','talk_time']]


# In[25]:


df.loc[(df.clock_speed >1.2) & (df.talk_time == '19')]


# In[26]:


df.loc[(df.clock_speed >2.2) & (df.n_cores == '5')]


# In[27]:


df.iloc[:7,:7]


# In[28]:


df[['clock_speed','int_memory','mobile_wt','ram']].sort_values(by='ram')


# In[29]:


df.query('41 < int_memory < 51').head()


# In[30]:


df.set_index('ram').tail()


# In[31]:


print(df.duplicated().sum())


# In[32]:


df.drop_duplicates(inplace=True)
print(df.duplicated().sum())


# In[33]:


import numpy as np
numerical_df = df.select_dtypes(include=[np.number])
numerical_df.head()


# In[34]:


categorical_df =df.select_dtypes(include='object')
categorical_df


# In[35]:


comb_df = pd.concat([numerical_df,categorical_df],axis=1)
comb_df.head(7)


# In[36]:


df["battery_power_qcut_bins"] = pd.qcut(df["battery_power"],q=5)
df["battery_power_qcut_bins"].value_counts()


# In[37]:


df["battery_power_cut_bins"] = pd.cut(df["battery_power"],bins=3,labels=["1-21","22-60","61-100"])
df["battery_power_cut_bins"].value_counts()


# In[38]:


df["battery_power_cut_bins"] = pd.cut(df["battery_power"],bins=3,labels=["1-21","22-60","61-100"])
df["battery_power_cut_bins"].value_counts()


# In[39]:


get_ipython().system('pip install pandas')


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[41]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"C:\python\mobile price project 2\train.csv")
display(df.head())


# In[42]:


df.describe()


# In[43]:


mean_value=df['ram'].mean()


# In[44]:


df=df.fillna('ram')


# In[47]:


print("replacing null values with mean")
df.isna().sum()


# In[48]:


df.duplicated().sum()


# In[49]:


y=df.px_width
x=df.drop('px_width',axis=1)


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)


# In[51]:


print("shape of a dataset",df.shape)


# In[52]:


print("shape of input-training set",x_train.shape)


# In[53]:


print("shape of output-training set",y_train.shape)


# In[54]:


print("shape of input-testing set",x_test.shape)


# In[55]:


print("shape of output-testing set",y_test.shape)


# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[57]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant, OLS


# In[58]:


df=pd.read_csv(r"C:\python\mobile price project 2\train.csv")
df


# In[59]:


df.shape


# In[64]:


df=pd.read_csv(r"C:\python\mobile price project 2\train.csv")
df


# In[65]:


new_df.describe()


# In[66]:


sns.distplot(new_df['mobile_wt'])


# In[67]:


new_df.describe().transpose() 


# In[68]:


new_df = df.copy() # copy your data to new data frame so what ever changes we will make don't affect to orignal. 
sns.boxplot(new_df['battery_power']) # dependent vaiable.


# In[72]:


hp = sorted(new_df['battery_power'])
q1, q3= np.percentile(hp,[25,75])
lower_bound = q1 -(1.5 * (q3-q1)) 
upper_bound = q3 + (1.5 * (q3-q1))
below = new_df['battery_power'] > lower_bound
above = new_df['battery_power'] < upper_bound
new_df = new_df[below & above]


# In[79]:


x = fullRaw2.drop(["battery_power"], axis = 1).copy()
y = fullRaw2["battery_power"].copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=100) 
    

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[80]:


from sklearn.linear_model import LinearRegression


# In[81]:


model = LinearRegression().fit(x_train,y_train)


# In[82]:


pred = model.predict(x_test)


# In[83]:


score1 = model.score(x_test,y_test) ## Co-effecient of determination (R - Square) to know how much it is fit #adjusted
score1


# In[84]:


1 - (1-model.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1) #adjusted R2 square


# In[85]:


fig, axes = plt.subplots(1, 1, sharex=False, sharey=False) #fig-->chart #axes 1,1 x,y #
fig.suptitle('[Residual Plots]') #supertitle
fig.set_size_inches(23,7) 
axes.plot(model.predict(x_test), y_test-model.predict(x_test), 'bo') #by which iam predicting bo defining line predict panra line 
#aa connect pandradhu
axes.axhline(y=1, color='m') #horizontal line
axes.grid()
axes.set_title('Linear')
axes.set_xlabel('predicted values')
axes.set_ylabel('residuals')


# In[86]:


import seaborn as sns

residuals_linear = y_test - model.predict(x_test)
sns.distplot(residuals_linear)
plt.title('Linear')


# In[87]:


predictors = x_train.columns

coef = pd.Series(model.coef_,predictors).sort_values() #coef from value it will predict

coef.plot(kind='bar', title='Model Coefficients')


# In[88]:


## Defining variables X,y 
X=df.drop("m_dep",axis=1) #axis column
y=df["m_dep"]
print("Columns in X :",X.columns)
print("y :",y)
print("shape of X:",X.shape)
print("shape of y:",y.shape[0])


# In[ ]:




