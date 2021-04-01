# Project Name:-Human Activity Recognition Using Smartphones


```python
#General useful packages
import pandas as pd
import numpy as np
#for visualisations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
```

**numpy and pandas:** Numpy enables us to work with arrays with great efficiency. Pandas is used to read the dataset file and import it as a dataframe, which is similar to a table with rows and columns.

**matplotlib:** Matplotlib is a highly customisable package which has the subpackage pyplot that enables us to draw plots, bar charts, pie charts and more. We get options to add legends, axis titles, change thickness of lines etc. The cm package (colormap) allows us to get colors for our charts.

**sklearn:** This machine learning library includes numerous machine learning algorithms already builtin with certain parameters set as default parameters, so they work right out of the box.


```python
#import all ML Algorithms
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#Accuracy score
from sklearn.metrics import accuracy_score
```

**Support Vector Machine**-Support vector machine (SVM) is supervised machine learning algorithm which can be used both for classification and regression problems. But generally, it is used in classification problems

**LogisticRegression**-It is a classification algorithm in machine learning that uses one or more independent variables to determine an outcome. The outcome is measured with a dichotomous variable meaning it will have only two possible outcomes. The goal of logistic regression is to find a best-fitting relationship between the dependent variable and a set of independent variables. 

**K-Nearest Neighbor**-It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space. It is a lazy learning algorithm as it does not focus on constructing a general internal model, instead, it works on storing instances of training data.

**Random Forest Classifier**-random forest are an ensemble learning method for classification, regression, etc. It operates by constructing a multitude of decision trees at training time and outputs the class that is the mode of the classes or classification or mean prediction(regression) of the individual trees.

**Accuracy score**-Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.

# Distinguish DataSet


```python
training_data=pd.read_csv('train.csv')
testing_data=pd.read_csv('test.csv')
```


```python
print('Training data')
training_data.shape
```

    Training data 




    (7352, 563)




```python
print("Null values present in training data:")
training_data.isnull().values.any()
```

    Null values present in training data:




    False




```python
print('Testing data')
testing_data.shape
```

    Testing data
    




    (2947, 563)




```python
print("Null values present in testing data:")
testing_data.isnull().values.any()
```

    Null values present in testing data:
    




    False




```python
training_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
      <th>subject</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 563 columns</p>
</div>




```python
testing_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
      <th>subject</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.257178</td>
      <td>-0.023285</td>
      <td>-0.014654</td>
      <td>-0.938404</td>
      <td>-0.920091</td>
      <td>-0.667683</td>
      <td>-0.952501</td>
      <td>-0.925249</td>
      <td>-0.674302</td>
      <td>-0.894088</td>
      <td>...</td>
      <td>-0.705974</td>
      <td>0.006462</td>
      <td>0.162920</td>
      <td>-0.825886</td>
      <td>0.271151</td>
      <td>-0.720009</td>
      <td>0.276801</td>
      <td>-0.057978</td>
      <td>2</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.286027</td>
      <td>-0.013163</td>
      <td>-0.119083</td>
      <td>-0.975415</td>
      <td>-0.967458</td>
      <td>-0.944958</td>
      <td>-0.986799</td>
      <td>-0.968401</td>
      <td>-0.945823</td>
      <td>-0.894088</td>
      <td>...</td>
      <td>-0.594944</td>
      <td>-0.083495</td>
      <td>0.017500</td>
      <td>-0.434375</td>
      <td>0.920593</td>
      <td>-0.698091</td>
      <td>0.281343</td>
      <td>-0.083898</td>
      <td>2</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.275485</td>
      <td>-0.026050</td>
      <td>-0.118152</td>
      <td>-0.993819</td>
      <td>-0.969926</td>
      <td>-0.962748</td>
      <td>-0.994403</td>
      <td>-0.970735</td>
      <td>-0.963483</td>
      <td>-0.939260</td>
      <td>...</td>
      <td>-0.640736</td>
      <td>-0.034956</td>
      <td>0.202302</td>
      <td>0.064103</td>
      <td>0.145068</td>
      <td>-0.702771</td>
      <td>0.280083</td>
      <td>-0.079346</td>
      <td>2</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.270298</td>
      <td>-0.032614</td>
      <td>-0.117520</td>
      <td>-0.994743</td>
      <td>-0.973268</td>
      <td>-0.967091</td>
      <td>-0.995274</td>
      <td>-0.974471</td>
      <td>-0.968897</td>
      <td>-0.938610</td>
      <td>...</td>
      <td>-0.736124</td>
      <td>-0.017067</td>
      <td>0.154438</td>
      <td>0.340134</td>
      <td>0.296407</td>
      <td>-0.698954</td>
      <td>0.284114</td>
      <td>-0.077108</td>
      <td>2</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.274833</td>
      <td>-0.027848</td>
      <td>-0.129527</td>
      <td>-0.993852</td>
      <td>-0.967445</td>
      <td>-0.978295</td>
      <td>-0.994111</td>
      <td>-0.965953</td>
      <td>-0.977346</td>
      <td>-0.938610</td>
      <td>...</td>
      <td>-0.846595</td>
      <td>-0.002223</td>
      <td>-0.040046</td>
      <td>0.736715</td>
      <td>-0.118545</td>
      <td>-0.692245</td>
      <td>0.290722</td>
      <td>-0.073857</td>
      <td>2</td>
      <td>STANDING</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 563 columns</p>
</div>



**We see there are 563 columns with the last column as Activity which will act as our label. In the remaining columns, subject is of no specific use for our machine learning application as we want to explore the activity and not who performed it. We can drop this column and the remaining 561 columns will be our features. Same will be done for the testing data.**


```python
#Training data variables
y_train=training_data['Activity']
X_train=training_data.drop(columns=['Activity','subject'])
```


```python
#Testing data variables
y_test = testing_data['Activity']
X_test = testing_data.drop(columns = ['Activity', 'subject'])
```

# Visualize the dataset

* We first get the count of records for each type of activity in a variable. value_counts() gives the counts.The counts are presented in alphabetical order of the name of the activity.

* unique() gives us unique values in y_train. We must sort it so that they align with the count values gathered above. The rcParams helps us define certain styles for out chart. We define the figure size using figure.figsize and the font size using font.size.

* We now have our data ready for the pie chart. pie() method creates a pie chart. The first argument is the count for each activity, the second argument is the respective activity name denoted by labels and the third argument autopct computes percentages for each activity.


```python
count_of_each_activity = np.array(y_train.value_counts())

activities = sorted(y_train.unique())

plt.rcParams.update({'figure.figsize': [20, 20], 'font.size': 24})
plt.pie(count_of_each_activity, labels = activities, autopct = '%0.2f')
```




    ([<matplotlib.patches.Wedge at 0xfc37fb0>,
      <matplotlib.patches.Wedge at 0xfc47410>,
      <matplotlib.patches.Wedge at 0xfc47810>,
      <matplotlib.patches.Wedge at 0xfc47c30>,
      <matplotlib.patches.Wedge at 0xfce9030>,
      <matplotlib.patches.Wedge at 0xfce9490>],
     [Text(0.9071064061014833, 0.6222201925441275, 'LAYING'),
      Text(-0.23874635466468208, 1.073778458591122, 'SITTING'),
      Text(-1.0745883152841482, 0.2350743555872831, 'STANDING'),
      Text(-0.7193129027755119, -0.832219290752544, 'WALKING'),
      Text(0.29301586483507763, -1.0602554894717366, 'WALKING_DOWNSTAIRS'),
      Text(1.0038008332903794, -0.4498709671511826, 'WALKING_UPSTAIRS')],
     [Text(0.4947853124189908, 0.3393928322967968, '19.14'),
      Text(-0.13022528436255384, 0.5856973410497028, '18.69'),
      Text(-0.5861390810640807, 0.12822237577488166, '17.49'),
      Text(-0.3923524924230064, -0.453937794955933, '16.68'),
      Text(0.15982683536458778, -0.5783211760754926, '14.59'),
      Text(0.5475277272492978, -0.24538416390064502, '13.41')])




    
![png](Images/output_18_1.png)
    


* **Next,I observe the type of readings in the datatset. If you look at the column headings, you can see that the columns have either the text refer to accelerometer reading,gyroscope values or none of the two to refer to all others.**

* **I first iterate through column names, to check if they contain ‘Acc’ or ‘Gyro’ or not. Based on the variable values, I plot a bar plot using bar() method of pyplot subpackage. The first argument are the X axis labels, the second argument takes the array of Y axis values and the color argument defines the colors red, blue and green respectively for the three bars. I again defined the figure size and font size for this bar plot**


```python
acc = 0
gyro = 0
others = 0
for column in X_train.columns:
    if 'Acc' in str(column):
        acc += 1
    elif 'Gyro' in str(column):
        gyro += 1
    else:
        others += 1
plt.rcParams.update({'figure.figsize': [10, 10], 'font.size': 16})
plt.bar(['Accelerometer', 'Gyroscope', 'Others'], [acc, gyro, others], color = ('r', 'b', 'g'))

```




    <BarContainer object of 3 artists>




    
![png](Images/output_20_1.png)
    


# Data Analysis

* The data collected is recorded at a stretch for each individual, especially for each activity. This means the records of any given activity will actually be in time series.

* I decided that I’d plot a line graph for all individuals who performed the Standing activity over a time period with respect to a feature. I took the feature as the angle between X and mean Gravity

**selecting all rows from the dataset that have the ‘Activity’ label as ‘STANDING’ and store it in standing_activity**


```python
standing_activity = training_data[training_data['Activity'] == 'STANDING']
# Reset the index for this dataframe
standing_activity = standing_activity.reset_index(drop=True)
```

**The data collected is in continuous time series for each individual and was recorded at the same rate. So, I can simply assign time values to each activity starting from '0' each time the subject changes. For each subject, the Standing activity records will start with a time value of 0 and increment by 1 till the previous row’s subject matches the present row’s subject.** 

**I store all the time series in a variable time_series and convert it into a dataframe using pandas method DataFrame() and store it in a variable time_series_df**



```python
time = 1
index = 0
time_series = np.zeros(standing_activity.shape[0])
for row_number in range(standing_activity.shape[0]):
    if (row_number == 0 
        or standing_activity.iloc[row_number]['subject'] == standing_activity.iloc[row_number - 1]['subject']):
        time_series[index] = time
        time += 1
    else:
        time_series[index] = 1
        time = 2
    index += 1

# Combine the time_series with the standing_activity dataframe
time_series_df = pd.DataFrame({ 'Time': time_series })
standing_activity_df = pd.concat([standing_activity, time_series_df], axis = 1)
```

* Now, as the data is ready to be plot, I use the matplotlib’s cm subpackage to get a list of colors using the rainbow metho

* iterate over the list of subjects inside the standing_activity_df. I specify size of the graph using rcParams. In the plot method, the first argument is X-axis values which is Time column in our case. The second column is for Y-axis values so I input the angle(X, gravityMean) values.


```python
colors = cm.rainbow(np.linspace(0, 1, len(standing_activity_df['subject'].unique())))

id = 0
for subject in standing_activity_df['subject'].unique():
    plt.rcParams.update({'figure.figsize': [40, 30], 'font.size': 24})
    plt.plot(standing_activity_df[standing_activity_df['subject'] == subject]['Time'], 
             standing_activity_df[standing_activity_df['subject'] == subject]['angle(X,gravityMean)'],
             c = colors[id], 
             label = 'Subject ' + str(subject),
             linewidth = 4)
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Angle between X and mean Gravity v/s Time for various subjects')
    plt.legend(prop = {'size': 24})
    id += 1
```


    
![png](Images/output_29_0.png)
    


# ML Algorithm
**Now comes the final step in our process to actually use machine learning algorithms and make classifications.**


```python
accuracy_scores = np.zeros(4)
```

**array of zeros of size 4 to store accuracy for each algorithm**

# SVM


```python
clf = SVC().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[0]))
```

    Support Vector Classifier accuracy: 95.04580929759076%
    

# Logistic Regression


```python
clf = LogisticRegression().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[1] = accuracy_score(y_test, prediction)*100
print('Logistic Regression accuracy: {}%'.format(accuracy_scores[1]))
```

    Logistic Regression accuracy: 95.79233118425518%
    

    d:\python\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

# KNN


```python
clf = KNeighborsClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[2] = accuracy_score(y_test, prediction)*100
print('K Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores[2]))
```

    K Nearest Neighbors Classifier accuracy: 90.02375296912113%
    

# Random Forest


```python
clf = RandomForestClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[3] = accuracy_score(y_test, prediction)*100
print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[3]))
```

    Random Forest Classifier accuracy: 92.46691550729555%
    

* **visualise the outputs as a bar graph.**


```python
colors = cm.rainbow(np.linspace(0, 1, 4))
labels = ['Support Vector Classifier', 'Logsitic Regression', 'K Nearest Neighbors', 'Random Forest']
plt.bar(labels,
        accuracy_scores,
        color = colors)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy of various algorithms')
```




    Text(0.5, 1.0, 'Accuracy of various algorithms')




    
![png](Images/output_42_1.png)
    


**we can see from the output values above, the Logistic Regression algorithm performed the best with the accuracy of over 96%.**

# Group members:
* Kushwanath Boina
* G Namrata Sai 
* MD. Jahid Hasan
* Simina Mannan Trisha


```python

```
