#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # #Reading the data set 

# In[2]:


Hotel_Bookings = pd.read_excel("C:/Users/Vidhut Sharma/Desktop/hotel_bookings_Capstone1.xlsx")
Hotel_Bookings.head(10)


# In[3]:


Hotel_Bookings[Hotel_Bookings["deposit_type"] == "Non Refund"]["is_canceled"].value_counts()


# # Checking the number of rows and column for the data

# In[4]:


Hotel_Bookings.shape


# # Checking the data types of the data and for the missing values and NA values

# In[5]:


Hotel_Bookings.info()


# In[6]:


#Na Values check
Hotel_Bookings.isnull().sum()


# Dropping the Features which are Country , Agent and Company as these are not the parameters which we can say will impactthe hotel cancellation and if we observe there are around 1 lakh + rows empty in the company column .If we remove that many NA values we may loose lot of our useful insights.

# In[7]:


Hotel_Bookings.drop(["country","agent","company"],axis=1,inplace = True)


# In[8]:


#Na Values check after removal of variables 
Hotel_Bookings.isnull().sum()


# In[9]:


Hotel_Bookings


# Now checking the difference between the booking date and arrival date. The difference will suggest us on how many days prior the arrival the customer books the hotel.
# 

# There are 94,428 rows in which there is the difference less than 180 days which we will be taking into consideration as anything above 180 days does not feel that secured that a person will not cancel the hotel

# In[10]:


Df_Of_Dates_Diff = pd.DataFrame({"booking_date": Hotel_Bookings["booking_date"],"arrival_date" : Hotel_Bookings['arrival_date'],"Difference":Hotel_Bookings['arrival_date'] - Hotel_Bookings["booking_date"]})
G = str(Df_Of_Dates_Diff["Difference"][0])
int(G.split()[0])
Dates_DiffList = []
for i in range(0,len(Df_Of_Dates_Diff["Difference"])):
    G = str(Df_Of_Dates_Diff["Difference"][i])
    Dates_DiffList.append(int(G.split()[0]))
Df_Of_Dates_Diff["Difference"] = Dates_DiffList


# Adding the Difference of dates in the Main Data of Hotel Cancellation

# In[11]:


Hotel_Bookings["Difference_Between_Booking_Arrival_Dates"] = Df_Of_Dates_Diff["Difference"]


# In[12]:


Hotel_Bookings["meal"].value_counts()


# # Starting the Univariate and Bivariate Analysis

# In[13]:


Hotel_Bookings["meal"].dtypes =="object"


# In[14]:


Hotel_Bookings["is_canceled"].unique()

for name in Hotel_Bookings["market_segment"].unique():
    fig = plt.figure(figsize = (7, 3))
    plt.bar(("Customers Not Cancelling","Customers Cancelling"),Hotel_Bookings[Hotel_Bookings["market_segment"]==name]["is_canceled"].value_counts(),color = "green")
    plt.title(name.upper() + " Value Seperation")
    plt.show()
#sns.barplot(x = Hotel_Bookings["is_canceled"].unique(),y = Hotel_Bookings[Hotel_Bookings["market_segment"]=="Online Travel Agents"]["is_canceled"].value_counts())


# In[15]:


Hotel_Bookings["is_canceled"].unique()

#for name in Hotel_Bookings["deposit_type"].unique():
fig = plt.figure(figsize = (7, 3))
plt.bar(("Customers Cancelling","Customers Not Cancelling"),Hotel_Bookings[Hotel_Bookings["deposit_type"]=="Non Refund"]["is_canceled"].value_counts(),color = "green")
plt.title("Non Refund")
plt.show()


# In[16]:


for name in Hotel_Bookings["customer_type"].unique():
    fig = plt.figure(figsize = (7, 3))
    plt.bar(("Customers Not Cancelling","Customers Cancelling"),Hotel_Bookings[Hotel_Bookings["customer_type"]==name]["is_canceled"].value_counts(),color = "green")
    plt.title(name.upper() + " Value Seperation")
    plt.show()


# In[17]:


for name in Hotel_Bookings["meal"].unique():
    fig = plt.figure(figsize = (7, 3))
    plt.bar(("Customers Not Cancelling","Customers Cancelling"),Hotel_Bookings[Hotel_Bookings["meal"]==name]["is_canceled"].value_counts(),color = "green")
    plt.title(name.upper() + " Value Seperation")
    plt.show()
    print(Hotel_Bookings[Hotel_Bookings["meal"]==name]["is_canceled"].value_counts(),sum(Hotel_Bookings[Hotel_Bookings["meal"]==name]["is_canceled"].value_counts()))
    


# In[18]:


for name in Hotel_Bookings.columns:
    if Hotel_Bookings[name].dtypes == "object":
        fig = plt.figure(figsize = (19, 3))
        plt.bar(Hotel_Bookings[name].unique(),Hotel_Bookings[name].value_counts(),color = "green",width =0.5)
        plt.title(name.upper() + " Value Seperation")
        plt.show()


# For the Bivariate Analysis we will be using the pairplot as it gives relationship with the dependent variable and independent variables and between independent and indepedent variable as well

# In[19]:


#sns.pairplot(Hotel_Bookings,aspect = 0.5);
#sns.PairGrid(Hotel_Bookings,aspect = 0.5)
Hotel_Data = sns.PairGrid(Hotel_Bookings,)
Hotel_Data = Hotel_Data.map(plt.scatter)

x_label,y_label = [],[]

for ax in Hotel_Data.axes[-1,:]:
    x_lab = ax.xaxis.get_label_text()
    x_label.append(x_lab)
for ax in Hotel_Data.axes[:,0]:
    y_lab = ax.yaxis.get_label_text()
    y_label.append(y_lab)

for i in range(len(x_label)):
    for j in range(len(y_label)):
        Hotel_Data.axes[j,i].xaxis.set_label_text(x_label[i])
        Hotel_Data.axes[j,i].yaxis.set_label_text(y_label[j])

plt.show()


# In[20]:


#Correalation Plot using heat map


# Checking for the Outliers present in the data set and we observe that the "stays_in_weekend_nights" and "stays_in_week_nights" have outliers present into them as compared to other features selected such as "adults" and "children" and "previous_cancellations" and "previous_bookings_not_canceled" 

# In[21]:


CheckForOutliers_Column_Selected = Hotel_Bookings[["stays_in_weekend_nights","stays_in_week_nights","adults","children","previous_cancellations","previous_bookings_not_canceled"]]
CheckForOutliers_Column_Selected
for name in CheckForOutliers_Column_Selected.columns:
    sns.boxplot(CheckForOutliers_Column_Selected[name])
    plt.show()


# In[22]:


CheckForOutliers_Column_Selected = Hotel_Bookings[["stays_in_weekend_nights","stays_in_week_nights"]]
for name in CheckForOutliers_Column_Selected.columns:
    sns.boxplot(CheckForOutliers_Column_Selected[name])
    plt.show()


# In[23]:


def remove_outliers_data(names):
    Q1 = Hotel_Bookings[names].describe()["25%"]
    Q3 = Hotel_Bookings[names].describe()["75%"]
    IQR = Q3 - Q1
    UL = Q3+ (1.5 * IQR)
    return UL


# In[24]:


for data in CheckForOutliers_Column_Selected.columns:
    Hotel_Bookings[data] = np.where(CheckForOutliers_Column_Selected[data] > remove_outliers_data(names = data) ,remove_outliers_data(data),CheckForOutliers_Column_Selected[data])


# Checking if outliers are removed from the features "stays_in_weekend_nights" and "stays_in_week_nights"

# In[25]:


CheckForOutliers_Column_Selected = Hotel_Bookings[["stays_in_weekend_nights","stays_in_week_nights"]]
for name in CheckForOutliers_Column_Selected.columns:
    sns.boxplot(Hotel_Bookings[name])
    plt.show()


# Transformation is required for the categorical variables as they have there categories in character format while modelling the variables should be converted to numrical categorical data for the model building  

# In[26]:


#Type Of Hotel
Hotel_Bookings["hotel"] = np.where(Hotel_Bookings["hotel"] == "type_1",0,Hotel_Bookings["hotel"])
Hotel_Bookings["hotel"] = np.where(Hotel_Bookings["hotel"] == "type_2",1,Hotel_Bookings["hotel"])
#Meal 
Hotel_Bookings["meal"] = np.where(Hotel_Bookings["meal"] == "Only Breakfast",0,Hotel_Bookings["meal"])
Hotel_Bookings["meal"] = np.where(Hotel_Bookings["meal"] == "Breakfast, lunch & dinner",1,Hotel_Bookings["meal"])
Hotel_Bookings["meal"] = np.where(Hotel_Bookings["meal"] == "Breakfast & dinner",2,Hotel_Bookings["meal"])
Hotel_Bookings["meal"] = np.where(Hotel_Bookings["meal"] == "No meal",3,Hotel_Bookings["meal"])
Hotel_Bookings["meal"].value_counts()

#Market_Segment
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Direct",0,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Corporate",1,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Online Travel Agents",2,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Offline Travel Agents/Operators",3,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Complementary",4,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Groups",5,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Undefined",6,Hotel_Bookings["market_segment"])
Hotel_Bookings["market_segment"] = np.where(Hotel_Bookings["market_segment"] == "Aviation",7,Hotel_Bookings["market_segment"])

#Distribution Channel
Hotel_Bookings["distribution_channel"] = np.where(Hotel_Bookings["distribution_channel"] == "Direct",0,Hotel_Bookings["distribution_channel"])
Hotel_Bookings["distribution_channel"] = np.where(Hotel_Bookings["distribution_channel"] == "Corporate",1,Hotel_Bookings["distribution_channel"])
Hotel_Bookings["distribution_channel"] = np.where(Hotel_Bookings["distribution_channel"] == "TA/TO",2,Hotel_Bookings["distribution_channel"])
Hotel_Bookings["distribution_channel"] = np.where(Hotel_Bookings["distribution_channel"] == "Undefined",3,Hotel_Bookings["distribution_channel"])
Hotel_Bookings["distribution_channel"] = np.where(Hotel_Bookings["distribution_channel"] == "GDS",4,Hotel_Bookings["distribution_channel"])

#Reserved Room Types
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "A",0,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "B",1,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "C",2,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "D",3,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "E",4,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "F",5,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "G",6,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "H",7,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "L",11,Hotel_Bookings["reserved_room_type"])
Hotel_Bookings["reserved_room_type"] = np.where(Hotel_Bookings["reserved_room_type"] == "P",15,Hotel_Bookings["reserved_room_type"])

#Assigned Room Type
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "A",0,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "B",1,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "C",2,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "D",3,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "E",4,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "F",5,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "G",6,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "H",7,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "I",8,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "K",10,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "L",11,Hotel_Bookings["assigned_room_type"])
Hotel_Bookings["assigned_room_type"] = np.where(Hotel_Bookings["assigned_room_type"] == "P",15,Hotel_Bookings["assigned_room_type"])

#Deposit Type
Hotel_Bookings["deposit_type"] = np.where(Hotel_Bookings["deposit_type"] == "No Deposit",0,Hotel_Bookings["deposit_type"])
Hotel_Bookings["deposit_type"] = np.where(Hotel_Bookings["deposit_type"] == "Refundable",1,Hotel_Bookings["deposit_type"])
Hotel_Bookings["deposit_type"] = np.where(Hotel_Bookings["deposit_type"] == "Non Refund",2,Hotel_Bookings["deposit_type"])

# Customer Type
Hotel_Bookings["customer_type"] = np.where(Hotel_Bookings["customer_type"] == "Transient",0,Hotel_Bookings["customer_type"])
Hotel_Bookings["customer_type"] = np.where(Hotel_Bookings["customer_type"] == "Contract",1,Hotel_Bookings["customer_type"])
Hotel_Bookings["customer_type"] = np.where(Hotel_Bookings["customer_type"] == "Transient-Party",2,Hotel_Bookings["customer_type"])
Hotel_Bookings["customer_type"] = np.where(Hotel_Bookings["customer_type"] == "Group",3,Hotel_Bookings["customer_type"])

Hotel_Bookings


# In[27]:


for name in Hotel_Bookings.columns:
    sns.boxplot(x = Hotel_Bookings["is_canceled"],y= Hotel_Bookings[name])
    plt.show()


# # Checking if the data set is unbalanced or not 
# 

# If the data is split in 60 - 40 or 50 -50 it is considered as good spread of data but in this case our data is split in approx. 63 - 37 which is kind of a ok split but it also has pinch oversampling.So to deal with it we might use oversampling method SMOTE

# In[28]:


sns.barplot(x = Hotel_Bookings["is_canceled"].unique(), y = Hotel_Bookings["is_canceled"].value_counts());
print("The percentage of customers who did not cancel their bookinsg are",Hotel_Bookings["is_canceled"].value_counts(normalize=True)[0],"%")
print("The percentage of customers who did cancel their bookinsg are",Hotel_Bookings["is_canceled"].value_counts(normalize=True)[1],"%")


# Clustering of data according to the difference between booking and arrival dates

# In[29]:


from sklearn.cluster import KMeans


# In[30]:


Hotel_Bookings_Without_Booking_Date_And_Arrival_Date = Hotel_Bookings.drop(["booking_date","arrival_date"],axis =1)
kmeans = KMeans(n_clusters= 3)


# In[31]:


kmeans.fit(Hotel_Bookings_Without_Booking_Date_And_Arrival_Date);


# In[32]:


kmeans.labels_
Hotel_Bookings_Without_Booking_Date_And_Arrival_Date["Cluster_Of_Data"] = kmeans.labels_


# Before balancing the data we should divide the data into training and testing data 
# 

# In[33]:


from sklearn.model_selection import train_test_split


# Firstly splitting the data in independent and Dependent variable

# In[34]:


X = Hotel_Bookings.drop(["is_canceled","booking_date","arrival_date","Difference_Between_Booking_Arrival_Dates"],axis =1)
y = Hotel_Bookings["is_canceled"]


# Second, now having the X and y we have to split the data reason for taking the random state in the train_test_split is that as we know our data is kind of imbalaced which can either give a lot of one category into train and less into test which can make our model predictions biased

# In[35]:


X_train,X_test,y_train,y_test = train_test_split(X , y ,test_size= 0.3,random_state= 43,stratify= y)


# Checking in X_train if there are any NA values after split are present or not 

# In[36]:


np.where(X_train.isna(),0,X_train.isna())


# Now starting with the Model building and first models which we will be taking into consideration will be Regression Models which are Linear and Logistics Regression. Importing regression models from sklearn package.

# In the case of regression models for checking if the model has performed good or not we use performance metrics which are RMSE[Root Mean Squared Error],MSE[Mean Squared Error],MAPE[Mean Absolute Percentage Error].As much these values are less the better the model performance

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error


# # Starting With Logistics Regression

# Taking the solver as Saga as it is considered to be good while we deal with huge data sets and with that we will be using stats.api package as well for the Logistic Regression Calculation to get other parameters such as R square and Adjsuted R Square as well

# In[38]:


LogReg = LogisticRegression(solver="saga")


# In[39]:


LogReg.fit(X_train , y_train)


# # Train Data Set Check for Logistics Regression

# In[40]:


LogReg.score(X_train, y_train)


# Mean Square Error , Root Mean Square Error , Mean Absolute Percentage Error

# In[41]:


Observed = y_train
Predicted = LogReg.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))

Log_RegDf = pd.DataFrame({"Model": "Logistic Regression(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Log_RegDf


# In[42]:


from sklearn.metrics import confusion_matrix , classification_report , roc_auc_score,roc_curve


# In[43]:


#prediction data for Train data set
y_train_pred = LogReg.predict(X_train)


# Confusion matrix , Classification Report , Roc_Curve, Roc_auc_curve

# In[44]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[45]:


print(classification_report(y_train,y_train_pred))


# In[46]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LR = pd.DataFrame({"Model": ["Logistics Regression(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LR


# In[47]:


probs = y_train_pred
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
train_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % train_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Logistics Regression

# In[48]:


Observed = y_test
Predicted = LogReg.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
Log_RegDf_test = pd.DataFrame({"Model": "Logistic Regression(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Log_RegDf_test 


# In[49]:


#prediction data for test data set
y_test_pred = LogReg.predict(X_test)


# In[50]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[51]:


print(classification_report(y_test,y_test_pred))


# In[52]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LR_Test = pd.DataFrame({"Model": ["Logistics Regression(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LR_Test


# In[53]:


probs = y_test_pred
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# In[54]:


probs[::1]


# # Starting With Linear Discriminant Analysis

# In[55]:


LDA = LinearDiscriminantAnalysis()


# In[56]:


LDA.fit(X_train,y_train)


# In[57]:


LDA.score(X_train, y_train)


# # Train Data Set Check for Linear Discriminant Analysis

# In[58]:


Observed = y_train
Predicted = LDA.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
LDA_train = pd.DataFrame({"Model": "Linear Discriminant Analysis(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
LDA_train


# In[59]:


#prediction data for Train data set
y_train_pred_LR = LDA.predict(X_train)


# In[60]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[61]:


print(classification_report(y_train,y_train_pred_LR))


# In[62]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LDA = pd.DataFrame({"Model": ["Linear Discriminant Analysis(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LDA


# In[63]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
train_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % train_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Linear Discriminant Analysis

# In[64]:


LDA.score(X_test, y_test)


# In[65]:


Observed = y_test
Predicted = LDA.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
LDA_test = pd.DataFrame({"Model": "Linear Discriminant Analysis(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
LDA_test


# In[66]:


#prediction data for test data set
y_test_pred_LR = LDA.predict(X_test)


# In[67]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[68]:


print(classification_report(y_test,y_test_pred_LR))


# In[69]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LDA_test = pd.DataFrame({"Model": ["Linear Discriminant Analysis(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LDA_test


# In[70]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# In[71]:


#Combining the data frame in Logistics and Linear Discrimant Analysis
Linear_Log_Df = pd.concat([Log_RegDf,Log_RegDf_test,LDA_train,LDA_test],axis =0)
Linear_Log_Df


# Classification Models 

# # Start Of Random Forest Classifier

# In[72]:


from sklearn.ensemble import RandomForestClassifier


# In[73]:


from sklearn.model_selection import GridSearchCV


# In[74]:


Rfcl = RandomForestClassifier()


# In[75]:


param_grid = {
    "max_depth" : [7,10],
    "max_features":[4,6],
    "min_samples_leaf":[50,100],
    "min_samples_split":[150,300],
    "n_estimators":[301,501]
}


# In[76]:


#grid_search = GridSearchCV(estimator=Rfcl , param_grid=param_grid , cv = 3)


# In[77]:


#grid_search.fit(X_train , y_train)


# In[78]:


#grid_search.best_params_


# In[79]:


Rfcl = RandomForestClassifier(n_estimators=501,min_samples_leaf=50,min_samples_split=150,max_depth=10,max_features=6)
Rfcl.fit(X_train,y_train)


# # Train Data Set Check for Random Forest Classifier

# In[80]:


Observed = y_train
Predicted = Rfcl.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
Rfcl_train = pd.DataFrame({"Model": "Random Forest Classifier(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Rfcl_train


# In[81]:


#prediction data for Train data set
y_train_pred_LR = Rfcl.predict(X_train)


# In[82]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[83]:


print(classification_report(y_train,y_train_pred_LR))


# In[84]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_RFCL = pd.DataFrame({"Model": ["Random Forest Classifier(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_RFCL


# In[85]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Random Forest Classifier

# In[86]:


Observed = y_test
Predicted = Rfcl.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
Rfcl_test = pd.DataFrame({"Model": "Random Forest Classifier(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Rfcl_test


# In[87]:


#prediction data for test data set
y_test_pred_LR = Rfcl.predict(X_test)


# In[88]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[89]:


print(classification_report(y_test,y_test_pred_LR))


# In[90]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_RFCL_test = pd.DataFrame({"Model": ["Random Forest Classifier(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_RFCL_test


# In[91]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of Decision Tree Classifier

# In[92]:


from sklearn.tree import DecisionTreeClassifier


# In[93]:


DT = DecisionTreeClassifier(criterion="gini",max_depth=13,min_samples_leaf=40,min_samples_split=20)
#DT = DecisionTreeClassifier(criterion="gini")
DT.fit(X_train,y_train)


# In[94]:


from sklearn import tree


# In[95]:


#train_char_labels = ["No" , "Yes"]
#Insurance_File = open("C:/Users/Vidhut Sharma/Documents/DataForPractice/Cap2.dot",mode = "w")
#dot_data = tree.export_graphviz(DT , out_file= Insurance_File , feature_names= list(X_train) , class_names=train_char_labels)


# # Train Data Set Check for Decision Tree Classifier

# In[96]:


Observed = y_train
Predicted = DT.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
DT_train = pd.DataFrame({"Model": "Decision Tree Classifier(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
DT_train


# In[97]:


#prediction data for Train data set
y_train_pred_LR = DT.predict(X_train)


# In[98]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[99]:


print(classification_report(y_train,y_train_pred_LR))


# In[100]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_DT = pd.DataFrame({"Model": ["Decision Tree Classifier(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_DT


# In[101]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Decision Tree Classifier

# In[102]:


Observed = y_test
Predicted = DT.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
DT_test = pd.DataFrame({"Model": "Decision Tree Classifier(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
DT_test


# In[103]:


#prediction data for test data set
y_test_pred_LR = DT.predict(X_test)


# In[104]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[105]:


print(classification_report(y_test,y_test_pred_LR))


# In[106]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_DT_test = pd.DataFrame({"Model": ["Decision Tree Classifier(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_DT_test


# In[107]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of MLP Classifier

# In[108]:


from sklearn.neural_network import MLPClassifier


# In[109]:


MLP = MLPClassifier()
MLP.fit(X_train,y_train)


# # Train Data Set Check for MLP Classifier

# In[110]:


Observed = y_train
Predicted = MLP.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
MLP_train = pd.DataFrame({"Model": "MLP Classifier(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
MLP_train


# In[111]:


#prediction data for Train data set
y_train_pred_LR = MLP.predict(X_train)


# In[112]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[113]:


print(classification_report(y_train,y_train_pred_LR))


# In[114]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_MLP = pd.DataFrame({"Model": ["MLP Classifier(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_MLP


# In[115]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for MLP Classifier

# In[116]:


Observed = y_test
Predicted = MLP.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
MLP_test = pd.DataFrame({"Model": "MLP Classifier(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
MLP_test


# In[117]:


#prediction data for test data set
y_test_pred_LR = MLP.predict(X_test)


# In[118]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[119]:


print(classification_report(y_test,y_test_pred_LR))


# In[120]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_MLP_test = pd.DataFrame({"Model": ["MLP Classifier(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_MLP_test


# In[121]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of AdaBoost Classifier

# In[122]:


from sklearn.ensemble import AdaBoostClassifier


# In[123]:


ABC = AdaBoostClassifier()
ABC.fit(X_train,y_train)


# # Train Data Set Check for AdaBoost Classifier

# In[124]:


Observed = y_train
Predicted = ABC.predict(X_train)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
ABC_train = pd.DataFrame({"Model": "Ada Boost Classifier(Train)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
ABC_train


# In[125]:


#prediction data for Train data set
y_train_pred_LR = ABC.predict(X_train)


# In[126]:


Confusion_MatrixForLogReg = confusion_matrix(y_train,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[127]:


print(classification_report(y_train,y_train_pred_LR))


# In[128]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_train,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_ABC = pd.DataFrame({"Model": ["Ada Boost Classifier(Train)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_ABC


# In[129]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_train, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for AdaBoost Classifier

# In[130]:


Observed = y_test
Predicted = ABC.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
ABC_test = pd.DataFrame({"Model": "Ada Boost Classifier(Test)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
ABC_test


# In[131]:


#prediction data for test data set
y_test_pred_LR = ABC.predict(X_test)


# In[132]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[133]:


print(classification_report(y_test,y_test_pred_LR))


# In[134]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_ABC_test = pd.DataFrame({"Model": ["Ada Boost Classifier(Test)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_ABC_test


# In[135]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Performace for all the metric calculated for all the models for train and test data set

# In[136]:


Total_MetricDf = pd.concat([Performance_Metric_LR, Performance_Metric_LR_Test, Performance_Metric_LDA, Performance_Metric_LDA_test, Performance_Metric_RFCL, Performance_Metric_RFCL_test, Performance_Metric_DT, Performance_Metric_DT_test, Performance_Metric_MLP, Performance_Metric_MLP_test, Performance_Metric_ABC, Performance_Metric_ABC_test],axis =0)
Metric_Df = pd.concat([Linear_Log_Df, Rfcl_train, Rfcl_test, DT_train, DT_test, MLP_train, MLP_test, ABC_train, ABC_test])
Total_MetricDf = pd.concat([Total_MetricDf,Metric_Df[["Mean Squared Error","Root Mean Squared Error","Mean Absolute Percentage Error"]]],axis =1)
Total_MetricDf
Total_MetricDf.rename({"Senstivity": "Sensitivity"},axis ="columns")
Total_MetricDf.columns = ["Model","Accuracy","Sensitivity","Precision","Mean Squared Error","Root Mean Squared Error","Mean Absolute Percentage Error"]
Total_MetricDf


# # Using Over Sampling Method SMOTE to make the data kind of less imbalanced 

# In[137]:


from imblearn.over_sampling import SMOTE


# In[138]:


sm = SMOTE(random_state=33,sampling_strategy=0.75)
X_res,y_res = sm.fit_sample(X_train,y_train)


# In[139]:


print(y_res.value_counts())#The Train values after SMOTE is used

print(y_train.value_counts())#Actual Train Values


# # Starting Logistics Regression (SMOTE)

# In[140]:


LogReg_SMOTE = LogisticRegression(solver="saga")
LogReg_SMOTE.fit(X_res , y_res)


# # Train Data For Logistics Regression(SMOTE)
# 

# In[141]:


Observed = y_res
Predicted = LogReg_SMOTE.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))

Log_RegDf_smote = pd.DataFrame({"Model": "Logistic Regression(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Log_RegDf_smote


# In[142]:


#prediction data for Train data set
y_train_pred = LogReg_SMOTE.predict(X_res)


# In[143]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[144]:


print(classification_report(y_res,y_train_pred))


# In[145]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LR_smote = pd.DataFrame({"Model": ["Logistic Regression(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LR_smote


# In[146]:


probs = y_train_pred
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
train_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % train_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data  for Logistics Regression(SMOTE)

# In[147]:


Observed = y_test
Predicted = LogReg_SMOTE.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))

Log_RegDf_smote_tst = pd.DataFrame({"Model": "Logistic Regression(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Log_RegDf_smote_tst


# In[148]:


#prediction data for Test data set
y_test_pred = LogReg_SMOTE.predict(X_test)


# In[149]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[150]:


print(classification_report(y_test,y_test_pred))


# In[151]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LR_smote_tst = pd.DataFrame({"Model": ["Logistic Regression(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LR_smote_tst


# In[152]:


probs = y_test_pred
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting Linear Discriminant Analysis (SMOTE)

# In[153]:


LDA= LinearDiscriminantAnalysis()


# In[154]:


LDA.fit(X_res,y_res)


# # Train Data Set Check for Linear Discriminant Analysis

# In[155]:


Observed = y_res
Predicted = LDA.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
LDA_train_smote = pd.DataFrame({"Model": "Linear Discriminant Analysis(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
LDA_train_smote


# In[156]:


#prediction data for Train data set
y_train_pred_LR = LDA.predict(X_res)


# In[157]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[158]:


print(classification_report(y_res,y_train_pred_LR))


# In[159]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LDA_smote = pd.DataFrame({"Model": ["Linear Discriminant Analysis(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LDA_smote


# In[160]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
train_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % train_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Linear Discriminant Analysis

# In[161]:


Observed = y_test
Predicted = LDA.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
LDA_test_smote = pd.DataFrame({"Model": "Linear Discriminant Analysis(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
LDA_test_smote


# In[162]:


#prediction data for test data set
y_test_pred_LR = LDA.predict(X_test)


# In[163]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[164]:


print(classification_report(y_test,y_test_pred_LR))


# In[165]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_LDA_test_smote = pd.DataFrame({"Model": ["Linear Discriminant Analysis(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_LDA_test_smote


# In[166]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# In[167]:


#Combining the data frame in Logistics and Linear Discrimant Analysis
Linear_Log_Df_smote = pd.concat([Log_RegDf_smote,Log_RegDf_smote_tst,LDA_train_smote,LDA_test_smote],axis =0)
Linear_Log_Df_smote


# # Start Of Random Forest Classifier(SMOTE)

# In[250]:


#Rfcl = RandomForestClassifier(n_estimators=301,min_samples_leaf=3,min_samples_split=10)
#Rfcl = RandomForestClassifier(n_estimators=501,min_samples_leaf=50,min_samples_split=150,max_depth=10,max_features=6)
Rfcl = RandomForestClassifier(n_estimators=501,min_samples_leaf=100,min_samples_split=80,max_depth=13,max_features=3)
Rfcl.fit(X_res,y_res) 


# # Train Data Set Check for Random Forest Classifier(SMOTE)

# In[251]:


Observed = y_res
Predicted = Rfcl.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
Rfcl_train_smote = pd.DataFrame({"Model": "Random Forest Classifier(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Rfcl_train_smote


# In[252]:


#prediction data for Train data set
y_train_pred_LR = Rfcl.predict(X_res)


# In[171]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[172]:


print(classification_report(y_res,y_train_pred_LR))


# In[253]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_RFCL_smote = pd.DataFrame({"Model": ["Random Forest Classifier(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_RFCL_smote


# In[174]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Random Forest Classifier(SMOTE)

# In[225]:


Observed = y_test
Predicted = Rfcl.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
Rfcl_test_smote = pd.DataFrame({"Model": "Random Forest Classifier(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
Rfcl_test_smote


# In[227]:


#prediction data for test data set
y_test_pred_LR = Rfcl.predict(X_test)


# In[177]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[178]:


print(classification_report(y_test,y_test_pred_LR))


# In[228]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_RFCL_test_smote = pd.DataFrame({"Model": ["Random Forest Classifier(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_RFCL_test_smote


# In[180]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of Decision Tree Classifier(SMOTE)

# In[181]:


DT = DecisionTreeClassifier(criterion="gini")
DT.fit(X_res,y_res)


# # Train Data Set Check for Decision Tree Classifier(SMOTE)

# In[182]:


Observed = y_res
Predicted = DT.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
DT_train_smote = pd.DataFrame({"Model": "Decision Tree Classifier(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
DT_train_smote


# In[183]:


#prediction data for Train data set
y_train_pred_LR = DT.predict(X_res)


# In[184]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[185]:


print(classification_report(y_res,y_train_pred_LR))


# In[186]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_DT_smote = pd.DataFrame({"Model": ["Decision Tree Classifier(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_DT_smote


# In[187]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for Decision Tree Classifier(SMOTE)

# In[188]:


Observed = y_test
Predicted = DT.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
DT_test_smote = pd.DataFrame({"Model": "Decision Tree Classifier(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
DT_test_smote


# In[189]:


#prediction data for test data set
y_test_pred_LR = DT.predict(X_test)


# In[190]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[191]:


print(classification_report(y_test,y_test_pred_LR))


# In[192]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_DT_test_smote = pd.DataFrame({"Model": ["Decision Tree Classifier(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_DT_test_smote


# In[193]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of MLP Classifier(SMOTE)

# In[194]:


MLP = MLPClassifier()
MLP.fit(X_res,y_res)


# # Train Data Set Check for MLP Classifier(SMOTE)

# In[195]:


Observed = y_res
Predicted = MLP.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
MLP_train_smote = pd.DataFrame({"Model": "MLP Classifier(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
MLP_train_smote


# In[196]:


#prediction data for Train data set
y_train_pred_LR = MLP.predict(X_res)


# In[197]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[198]:


print(classification_report(y_res,y_train_pred_LR))


# In[199]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_MLP_smote = pd.DataFrame({"Model": ["MLP Classifier(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_MLP_smote


# In[200]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for MLP Classifier(SMOTE)

# In[201]:


Observed = y_test
Predicted = MLP.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
MLP_test_smote = pd.DataFrame({"Model": "MLP Classifier(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
MLP_test_smote


# In[202]:


#prediction data for test data set
y_test_pred_LR = MLP.predict(X_test)


# In[203]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[204]:


print(classification_report(y_test,y_test_pred_LR))


# In[205]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_MLP_test_smote = pd.DataFrame({"Model": ["MLP Classifier(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_MLP_test_smote


# In[206]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Starting of AdaBoost Classifier(SMOTE)

# In[207]:


ABC = AdaBoostClassifier()
ABC.fit(X_res,y_res)


# # Train Data Set Check for AdaBoost Classifier(SMOTE)

# In[208]:


Observed = y_res
Predicted = ABC.predict(X_res)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
ABC_train_smote = pd.DataFrame({"Model": "Ada Boost Classifier(Train For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
ABC_train_smote


# In[209]:


#prediction data for Train data set
y_train_pred_LR = ABC.predict(X_res)


# In[210]:


Confusion_MatrixForLogReg = confusion_matrix(y_res,y_train_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[211]:


print(classification_report(y_res,y_train_pred_LR))


# In[212]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_res,y_train_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_ABC_smote = pd.DataFrame({"Model": ["Ada Boost Classifier(Train For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_ABC_smote


# In[213]:


probs = y_train_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
traun_auc = roc_auc_score(y_res, probs)
print('AUC: %.3f' % traun_auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_res, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# # Test Data Set Check for AdaBoost Classifier(SMOTE)

# In[214]:


Observed = y_test
Predicted = ABC.predict(X_test)

mse = np.mean((Observed-Predicted)**2)
print("The Mean Squared Error is:",mse)
rmse = sqrt(mse)
print("The Root Mean Squared Error is:",rmse)
print("The Mean Absolute Percentage Error is:",mean_absolute_percentage_error(Observed,Predicted))
ABC_test_smote = pd.DataFrame({"Model": "Ada Boost Classifier(Test For SMOTE)","Mean Squared Error":[mse],"Root Mean Squared Error": [rmse],"Mean Absolute Percentage Error":[mean_absolute_percentage_error(Observed,Predicted)]})
ABC_test_smote


# In[215]:


#prediction data for test data set
y_test_pred_LR = ABC.predict(X_test)


# In[216]:


Confusion_MatrixForLogReg = confusion_matrix(y_test,y_test_pred_LR)

names_on_cm = ["True Neg","False Pos","False Neg","True Pos"]
names_percentages = ["{0:.2%}".format(value_data) for value_data in
                     Confusion_MatrixForLogReg.flatten()/np.sum(Confusion_MatrixForLogReg)]
labels = [f"{v1}\n{v3}" for v1, v3 in
          zip(names_on_cm,names_percentages)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Confusion_MatrixForLogReg, annot=labels, fmt="", cmap='Greens')


# In[217]:


print(classification_report(y_test,y_test_pred_LR))


# In[218]:


#Data Frame Creation for Accuracy , Precision and Recall
Confusion_MatrixData = confusion_matrix(y_test,y_test_pred_LR)

Accuracy = (Confusion_MatrixData[1][0] + Confusion_MatrixData[0][0])/(Confusion_MatrixData[0][0] + Confusion_MatrixData[0][1] + Confusion_MatrixData[1][0]+ Confusion_MatrixData[1][1])
Senstivity = (Confusion_MatrixData[1][1]) /  (Confusion_MatrixData[1][1] + Confusion_MatrixData[1][0])
Precision = (Confusion_MatrixData[1][1]) / (Confusion_MatrixData[1][1] + (Confusion_MatrixData[0][1])) 

Performance_Metric_ABC_test_smote = pd.DataFrame({"Model": ["Ada Boost Classifier(Test For SMOTE)"],"Accuracy" : [Accuracy],"Senstivity":[Senstivity],"Precision":[Precision]})
Performance_Metric_ABC_test_smote


# In[219]:


probs = y_test_pred_LR
# keep probabilities for the positive outcome only
probs = probs
# calculate AUC
test_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# # Performace for all the metric calculated for all the models for train and test data set (SMOTE)

# In[220]:


Total_MetricDf_Smote = pd.concat([Performance_Metric_LR_smote, Performance_Metric_LR_smote_tst, Performance_Metric_LDA_smote, Performance_Metric_LDA_test_smote, Performance_Metric_RFCL_smote, Performance_Metric_RFCL_test_smote, Performance_Metric_DT_smote, Performance_Metric_DT_test_smote, Performance_Metric_MLP_smote, Performance_Metric_MLP_test_smote, Performance_Metric_ABC_smote, Performance_Metric_ABC_test_smote],axis =0)
Metric_Df_SMOTE = pd.concat([Log_RegDf_smote, Log_RegDf_smote_tst, LDA_train_smote, LDA_test_smote, Rfcl_train_smote, Rfcl_test_smote, DT_train_smote, DT_test_smote, MLP_train_smote, MLP_test_smote, ABC_train_smote, ABC_test_smote])
Total_MetricDf_Smote = pd.concat([Total_MetricDf_Smote,Metric_Df_SMOTE[["Mean Squared Error","Root Mean Squared Error","Mean Absolute Percentage Error"]]],axis =1)
Total_MetricDf_Smote
Total_MetricDf_Smote.rename({"Senstivity": "Sensitivity"},axis ="columns")
Total_MetricDf_Smote.columns = ["Model","Accuracy","Sensitivity","Precision","Mean Squared Error","Root Mean Squared Error","Mean Absolute Percentage Error"]
Total_MetricDf_Smote


# In[ ]:




