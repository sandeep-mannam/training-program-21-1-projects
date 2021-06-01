# Car Price Prediction

<img src="http://www.vehiclemix.com/wp-content/uploads/2014/06/thinking_man.png">

#### Mentor
<ul type="square">
  <li><b><a href="https://www.linkedin.com/in/samiksha-bhavsar-33837417a/">Samiksha Bhavsar</a></b><br></li>
  <li><b><a href="https://www.linkedin.com/in/giduturi-namrata-0898991b1/">Giduturi Namrata Sai</a></b></li>
</ul>

#### Members

||Name|
|-|-|
|<li>|<a href="https://www.linkedin.com/in/hemachandiran-t-081836171/">Hemachandiran</a>|
|<li>|<a href="https://www.linkedin.com/in/sagar-dhandare-a401271a3/">Sagar Dhandare P ML P</a>|
|<li>|<a href="https://www.linkedin.com/in/sourav-pahwa-93b4041b6/">Sourav Pahwa</a>|
|<li>|<a href="https://www.linkedin.com/in/nikita-srivastava-0738bb162/">Nikita Srivastava</a>|


#### About Project 
In the Car Price Prediction Project, it is about predicting price of the car on the basis of different variables used for prediction as follows:

<li>Present_Price</li>
<li>Kms_Driven</li>
<li>Fuel_Type</li>
<li>Seller_Type</li>
<li>Transmission</li>
<li>Owner</li>
<li>Number_Of_Years</li>

<br>
<b>You can download the dataset from <a href="https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv">here</a></b>
<br><br>
We have used various libraries such as <b>Numpy</b>, <b>Pandas</b>, <b>Scikit-learn</b>, <b>Matplotlib</b> and <b>Seaborn</b>. 
<br><br>
<b>Numpy and pandas</b> were used for <b>analysing the data</b>, <b>Scikit-learn</b> was used for <b>Mathematical Computations</b> and <b>Matplotlib and Seaborn</b> are used for <b>data visualization</b> i.e. for plotting different types of Graphs.

#### Workflow

<ol>
  <li> We downloaded the dataset from kaggle and then imported the dataset.</li>
  <li>Then we have started performing EDA on dataset i.e. data preprocessing, data cleaning and data transformation.</li>
  <li>Firstly, we checked for any null values present in the dataset and then dropped the columns containing those null values!</li>
  <li>Secondly, we have plotted the correlation between the variables using heatmap and pairplot both from seaborn library!!</li>
  <li>Then we used various ML Algortihms like <b>Random Forest Regressor</b> (A Supervised Learning Algorithm) and <b>Grid Seacrh CV </b>as an Optimiztion Algorithm!!!</li>
  <li>At Last, We have printed relation between Selling Price and Count using Histogram and Scatter Plot and also printed <b>Mean Absolute Error(MAE), Mean Squared Error(MSE)</b> and </b>Root Mean Square Error (RMSE)</b></li>
  <li><b>At last we have seen the R2 score</b> (proportion of data points which lie within the line created by the regression equation) <b>using metrics imported from sklearn library. We have gotten 0.8691 as R2 score which means our model is accurate.</b></li>

</ol>
