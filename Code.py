# **1. Importing Libraries**

# manipulation data
import pandas as pd
import numpy as np

#visualiation data
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import warnings
from scipy import stats
from scipy.stats import norm

#default theme
plt.style.use('ggplot')
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'

# **2. Reading Datasets**

from google.colab import drive
drive.mount("/content/drive")

df1=pd.read_csv("/content/drive/MyDrive/train.csv")
df2=pd.read_csv("/content/drive/MyDrive/test.csv")


#**3. Data Preprocessing**

df1.head()

df1. tail()

df1.shape

df1.isnull().sum()

df1.dtypes

df1.dtypes.value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.2f%%',shadow=True)
plt.title('type of our data');

df1.info()

df2.dtypes

df2.dtypes.value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.2f%%',shadow=True)
plt.title('type of our data');

df1.describe(include='all')

A basic observation is that:

*   Product P00265242 is the most popular product.
*   Most of the transactions were made by men.
*   Age group with most transactions was 26-35.
*   City Category with most transactions was B 




**Correlation Matrix**

corr = df1.corr()
plt.figure(figsize=(30,10))
sns.heatmap(corr, annot=True)

**Finding Missing Values**

missing_values=df1.isnull().sum()
percent_missing = df1.isnull().sum()/df1.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame

missing_values = df1.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
missing_values.plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
plt.title('our missing values');

Only Product_Category_2 and Product_Category_3 have null values. However Product_Category_3 is null for nearly 70% of transactions so it can't give us much information. so we are dropping Product_Category_3

**For Product_Category_2**

df1.Product_Category_2.value_counts()

df1.Product_Category_2.describe()

# Replace using median 
median = df1['Product_Category_2'].median()
df1['Product_Category_2'].fillna(median, inplace=True)

**For Product_Category_3**

df1.Product_Category_3.value_counts()

# drop Product_Category_3 
df1=df1.drop('Product_Category_3',axis=1)

**Re-Checking the Missing Values**

missing_values=df1.isnull().sum()
percent_missing = df1.isnull().sum()/df1.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame

# **4. Data Analysis and Data Handling**

df1.hist(edgecolor='black',figsize=(12,12));

df1.columns

**Gender**

# pie chart 

size = df1['Gender'].value_counts()
labels = ['Male', 'Female']
colors = ['#C4061D', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('A Pie Chart representing the gender gap', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


sns.countplot(x=df1.Gender)
plt.title('Gender per transaction');

**Age**


ageData = sorted(list(zip(df1.Age.value_counts().index, df1.Age.value_counts().values)))
age, productBuy = zip(*ageData)
age, productBuy = list(age), list(productBuy)
ageSeries = pd.Series((i for i in age))

data = [go.Bar(x=age, 
               y=productBuy, 
               name="How many products were sold",
               marker = dict(color=['black', 'yellow', 'green', 'blue', 'red', 'gray', '#C4061D'],
                            line = dict(color='#7C7C7C', width = .5)),
              text="Age: " + ageSeries)]
layout = go.Layout(title= "How many products were sold by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

**Occupation of Customers**

palette=sns.color_palette("Set2")
plt.rcParams['figure.figsize'] = (18, 9)
sns.countplot(df1['Occupation'], palette = palette)
plt.title('Distribution of Occupation across customers', fontsize = 20)
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()
spent_by_occ = df1.groupby(by='Occupation').sum()['Purchase']
plt.figure(figsize=(20, 7))
sns.barplot(x=spent_by_occ.index,y=spent_by_occ.values)
plt.title('Total Money Spent per Occupation')
plt.xlabel('Occupation')
plt.ylabel('Money_spent')
plt.show()

Once again, the distribution of the mean amount spent within each occupation appears to mirror the distribution of the amount of people within each occupation. This is fortunate from a data science perspective, as we are not working with odd or outstanding features. Our data, in terms of age and occupation seems to simply make sense.

**City Category**

plt.rcParams['figure.figsize'] = (18, 9)
sns.countplot(df1['City_Category'], palette = palette)
plt.title('Distribution of Cities across customers', fontsize = 20)
plt.xlabel('Cities')
plt.ylabel('Count')
plt.show()


**Stay In Current City Years**

df1['Stay_In_Current_City_Years'].replace('4+', 4, inplace = True)
sns.countplot(df1['Stay_In_Current_City_Years'], palette = 'copper')
plt.title('Distribution of Stay across customers', fontsize = 20)
plt.xlabel('Distribution of Stay')
plt.ylabel('Count')
plt.show()

**Products**

plt.figure(figsize=(20,6))
prod_by_cat = df1.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index,y=prod_by_cat.values, palette=palette)
plt.title('Number of Unique Items per Category')
plt.show()

**Category labels 1, 5, and 8 clearly have the most items within them. This could mean the store is known for that item, or that the category is a broad one.**

category = []
mean_purchase = []


for i in df1['Product_Category_1'].unique():
    category.append(i)
category.sort()

for e in category:
    mean_purchase.append(df1[df1['Product_Category_1']==e]['Purchase'].mean())

plt.figure(figsize=(20,6))

sns.barplot(x=category,y=mean_purchase)
plt.title('Mean of the Purchases per Category')
plt.xlabel('Product Category')
plt.ylabel('Mean Purchase')
plt.show()

# visualizing the different product categories

plt.rcParams['figure.figsize'] = (15, 25)
plt.style.use('ggplot')

plt.subplot(4, 1, 1)
sns.countplot(df1['Product_Category_1'], palette = palette)
plt.title('Product Category 1', fontsize = 20)
plt.xlabel('Distribution of Product Category 1')
plt.ylabel('Count')

plt.subplot(4, 1, 2)
sns.countplot(df1['Product_Category_2'], palette = palette)
plt.title('Product Category 2', fontsize = 20)
plt.xlabel('Distribution of Product Category 2')
plt.ylabel('Count')


plt.show()

**The Purchase Attribute Which Is Our Target Variable**

# plotting a distribution plot for the target variable
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 7)
sns.distplot(df1['Purchase'], color = 'green', fit = norm)

# fitting the target variable to the normal curve 
mu, sigma = norm.fit(df1['Purchase']) 
print("The mu {} and Sigma {} for the curve".format(mu, sigma))

plt.title('A distribution plot to represent the distribution of Purchase')
plt.legend(['Normal Distribution ($mu$: {}, $sigma$: {}'.format(mu, sigma)], loc = 'best')
plt.show()

# **5. Data Processing 2**


**Removing unwanted columns**

# saving the attributes User_ID and Product_ID before deleting them
User_ID = df1['User_ID']
Product_ID = df1['Product_ID']

df1 = df1.drop(['User_ID', 'Product_ID'], axis = 1)


# checking the new shape of data
df1.shape

**Label encoding**

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['Gender'] = le.fit_transform(df1['Gender'])
df1['City_Category'] = le.fit_transform(df1['City_Category'])
df1['Age'] = le.fit_transform(df1['Age'])
df1['Stay_In_Current_City_Years'] = le.fit_transform(df1['Stay_In_Current_City_Years'])
df1

**Splitting the data into dependent and independents sets**

y = df1['Purchase']

# now removing the purchase column from the dataset
df1 = df1.drop(['Purchase'], axis = 1)

x = df1

# checking the shapes of x and y
print("Shape of x: ", x.shape)
print("Shape of y: ", y.shape)

**Splitting into training and testing**

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 15)

print("Shape of x_train: ", x_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ",y_test.shape)

**Standardization**

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# **6. Modelling**

**Ridge Regression**

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import *

model = Ridge()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# finding the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

**Elastic Net Regression**

from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import *

model = ElasticNet()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# finding the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

**Lasso Regression**


from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import *

model = Lasso()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# finding the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

**Gradient Boosting Regression**

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import *

model = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# finding the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

**Comparison of R2 Score**

r2_score = np.array([0.12736140409143537, 0.11452566055708235, 0.12735819037238216, 0.653185708142229])
labels = np.array(['Ridge', 'Elastic', 'Lasso', 'Gradient Boosting'])
indices = np.argsort(r2_score)
color = plt.cm.rainbow(np.linspace(0, 1, 4))

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18, 7)
plt.bar(range(len(indices)), r2_score[indices], color = color)
plt.xticks(range(len(indices)), labels[indices])
plt.title('R2 Score', fontsize = 25)
plt.grid()
plt.tight_layout()
plt.show()

**Comparison of RMSE Score**

rmse_score = np.array([4703.496079726382, 4737.961939753273, 4703.504740643469, 2965.186885509144])
labels = np.array(['Ridge', 'Elastic', 'Lasso', 'Gradient Boosting'])
indices = np.argsort(r2_score)
color = plt.cm.rainbow(np.linspace(0, 1, 6))

plt.style.use('seaborn-talk')
plt.rcParams['figure.figsize'] = (18, 7)
plt.bar(range(len(indices)), rmse_score[indices], color = color)
plt.xticks(range(len(indices)), labels[indices])
plt.title('RMSE Score', fontsize = 30)
plt.grid()
plt.tight_layout()
plt.show()

# **7. Result**

median = df2['Product_Category_2'].median()
df2['Product_Category_2'].fillna(median, inplace=True)
df2 = df2.drop('Product_Category_3',axis=1)

model = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred.astype(int)





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df2['Gender'] = le.fit_transform(df2['Gender'])
df2['City_Category'] = le.fit_transform(df2['City_Category'])
df2['Age'] = le.fit_transform(df2['Age'])
df2['Stay_In_Current_City_Years'] = le.fit_transform(df2['Stay_In_Current_City_Years'])

test = df2.drop(columns=["User_ID",'Product_ID'])
test

pred_test=model.predict(test)
Submission=pd.read_csv("/content/drive/MyDrive/sample_submission.csv")

Submission["Purchase"]=pred_test
Submission[["User_ID","Product_ID"]]=df2[["User_ID","Product_ID"]]
Submission

pd.DataFrame(Submission, columns=["Purchase","User_ID","Product_ID"]).to_csv(
    r"/content/drive/MyDrive/Submission.csv", index=False)
