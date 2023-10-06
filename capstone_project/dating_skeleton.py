# Using this as a basis: https://github.com/AliPakzad/Machine-Learning-Capstone-Project/blob/main/date-a-scientist.ipynb
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###########################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error


#############################################

#Create your df here:
df = pd.read_csv("profiles.csv")

#####################
# normalise the dataset counts by male/female instead of using counts
#####################

#print(df.head(15))
#print(df.pets.head())
#print(df.info())
#print(df.describe())
#print(df.shape())   # doesn't work here!

#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.show()

#val = df.sign.value_counts()
#print(val)
#sns.displot(data=df, x="height", hue="sex", binwidth=5, multiple="stack")
#plt.xlim(50,85)

'''sns.displot(data=df, x="income")
plt.xlim(0,1e6)
plt.ylim(0,0.6e5)
plt.show()'''
#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="body_type", hue="sex")
#sns.countplot(data=df, x="diet", hue="sex")     #females more likely for vegetarian/vegan
#plt.xticks(rotation=90)

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, x="drinks", hue="sex")     
#sns.countplot(data=df, x="drugs", hue="sex")     
#sns.countplot(data=df, y="smokes", hue="sex") 

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, x="education", hue="sex")     

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, x="job", hue="sex")  
#plt.xticks(rotation=90)
#plt.show()

#print(df.job.value_counts())

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="offspring", hue="sex")  

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="orientation", hue="sex")  

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="pets", hue="sex") 

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="religion", hue="sex") 
# order by religion groups, no separation on how serious. Loads of no answer, NaN.
df['religion_cleaned'] = df['religion'].str.split().str[0]
#sns.countplot(data=df, y="religion_cleaned", hue="sex") 
# clean this data
df['sign_cleaned'] = df['sign'].str.split().str[0]

#plt.figure(figsize=(8,6))
#sns.countplot(data=df, y="sign_cleaned", hue="sex") 

#plt.figure(figsize=(8,6))
#plt.show()

#print(df.isnull().sum())


drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smoke_mapping)
drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drug_mapping)
body_type_codes = {'thin':0, 'skinny':1, 'fit':2, 'athletic':3, 'jacked':4, 'rather not say':5, 'average':6, 'a little extra':7, 'used up':8, 'curvy':9, 'overweight':10, 'full figured':11}
df["body_type_code"] = df['body_type'].map(body_type_codes)
#print(df["body_type_code"])
#print(df["body_type_code"].value_counts())
#print(df.drugs.value_counts())
#print(df.religion.value_counts())
#print(df.religion.info())

##essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
'''all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df['word_len'] = all_essays.apply(lambda x: len(x.split()))

#word_len = all_essays.str.split().apply(lambda x: [len(i) for i in x]).mean()
#df['avg_word_length'] = df['word_len'].mean()
#print(word_len)

df["essay_len"] = all_essays.apply(lambda x: len(x))
##df["avg_word_len"] = all_essays.apply(sum(len(word) for word in all_essays) / len(all_essays))
#print(all_essays[3])'''

# get only selected items for data anaylsis, and remove any NaN values while we're at it
grabbed = ['sign_cleaned', 'body_type_code', 'diet', 'orientation', 'pets', 'religion_cleaned', 'sex', 'job', 'smokes_code', 'drinks_code' , 'drugs_code']
data = df[grabbed].dropna()
#data = data.dropna()

# convert diet and job into dummy indicator vars
dummy_list = ["diet", "job", "orientation", "pets", "religion_cleaned", "sex", "body_type_code"]
for item in dummy_list:
    data = pd.get_dummies(data, columns=[item], prefix=[item])
#print(data.head(15))


############### Split the data
features = data.iloc[:, 1:]
label = data['sign_cleaned']

X_train, X_Test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=1)

y_train = y_train.ravel()
y_test = y_test.ravel()

#scale and normalise the data
scaler = MinMaxScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_Test)

################## Build the models
##### Naive Bayes
'''NbModel = MultinomialNB()

NbModel.fit(X_train_Scaled, y_train)
accuracy = NbModel.score(X_train_Scaled, y_train)

#print("accuracy is: ", str(accuracy))
predictions = NbModel.predict(X_test_scaled)
result = classification_report(y_test, predictions)'''
#print("\nPredictions: ", str(result))

##### Logistic Regression
'''logReg = LogisticRegression(multi_class="multinomial")
logReg.fit(X_train_Scaled, y_train)
logRegAcc = logReg.score(X_train_Scaled, y_train)
print("accuracy is: ", str(logRegAcc))

logReg_pred = logReg.predict(X_test_scaled)
logRegRes = classification_report(y_test, logReg_pred)
print("\nPredictions: ", str(logRegRes))'''

##### K Nearest Neighbour
# Find best K
'''KNN_results = []
kMax = 80
kList = list(range(1,kMax+1))

for k in range(1,kMax+1):
    KNNmodel = KNeighborsClassifier(n_neighbors = k)
    KNNmodel.fit(X_train_Scaled, y_train)
    KNN_results.append(KNNmodel.score(X_test_scaled, y_test))
    
plt.plot(kList, KNN_results)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.show()

# show results from model using best k of 20 - try to get best k automatically to feed in here instead of manually!
KNNmodel = KNeighborsClassifier(n_neighbors = 20)
KNNmodel.fit(X_train_Scaled, y_train)
knn_acc = KNNmodel.score(X_train_Scaled, y_train)

print("KNN accuracy is: {}%".format(round(knn_acc, 2) *100))
predictions = KNNmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''


##### Decision Tree Classifier
'''DT_model = DecisionTreeClassifier()

DT_model.fit(X_train_Scaled, y_train)
DT_acc = DT_model.score(X_train_Scaled, y_train)

print("DT accuracy is: {}%".format(round(DT_acc, 2) *100))'''

#DT_pred = DT_model.predict(X_test_scaled)
#print(classification_report(y_test, DT_pred))

######################
#Take religion out and see if DT performs any better
# get only selected items for data anaylsis, and remove any NaN values while we're at it
'''grabbed2 = ['sign_cleaned', 'body_type_code', 'diet', 'orientation', 'pets', 'sex', 'job', 'smokes_code', 'drinks_code' , 'drugs_code']
data2 = df[grabbed2].dropna()

# convert diet and job into dummy indicator vars
dummy_list2 = ["diet", "job", "orientation", "pets", "sex", "body_type_code"]
for item in dummy_list2:
    data2 = pd.get_dummies(data2, columns=[item], prefix=[item])
#print(data.head(15))

features2 = data2.iloc[:, 1:]
label2 = data2['sign_cleaned']

X_train2, X_Test2, y_train2, y_test2 = train_test_split(features2, label2, test_size=0.3, random_state=1)

y_train2 = y_train2.ravel()
y_test2 = y_test2.ravel()

#scale and normalise the data
scaler2 = MinMaxScaler()
X_train_Scaled2 = scaler2.fit_transform(X_train2)
X_test_scaled2 = scaler2.fit_transform(X_Test2)

#model on new data
DT_model2 = DecisionTreeClassifier()

DT_model2.fit(X_train_Scaled2, y_train2)
DT_acc2 = DT_model2.score(X_train_Scaled2, y_train2)

print("new DT accuracy is: {}%".format(round(DT_acc2, 2) *100))
DT_pred = DT_model.predict(X_test_scaled)
print(classification_report(y_test, DT_pred))'''

# Result is:
# reduction of trained accuracy, and a slight reduction of overall accuracy, but is negligible.

##############################
# question 2
# can we predict body type using lifestyle information?
'''q2data = df
grabbed = ['body_type', 'diet', 'sex', 'orientation', 'job', 'smokes_code', 'drinks_code' , 'drugs_code', 'age', 'pets']
q2data = df[grabbed].dropna()

# convert diet and job into dummy indicator vars
dummy_list2 = ["diet", "job", "orientation", "sex", 'pets']
for item in dummy_list2:
    q2data = pd.get_dummies(q2data, columns=[item], prefix=[item])

#print(q2data.shape)
#print(q2data.head(15))

#Y is the target column, X has the features
X = q2data.iloc[:, 1:]
y = q2data['body_type']


#Split the data into training set and test set and normalise it all
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

#Pandas Series.ravel() function returns the flattened underlying data as an ndarray(1d array)
y_train = y_train.ravel()
y_test = y_test.ravel()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)'''

#########
# Perform regression on the data
'''logRegModel = LogisticRegression(multi_class= "multinomial")
logRegModel.fit(X_train_scaled, y_train)

trainingAcc = logRegModel.score(X_train_scaled, y_train)
print("Q3 accuracy of model for training data: {}%".format(round(trainingAcc, 2) *100))

logRegPred = logRegModel.predict(X_test_scaled)
print(classification_report(y_test, logRegPred))
'''
# with state of 40, model is 31% accurate, and 31% accurate on real data


#########
# KNN data model
'''Knn_Acc = []
kMax = 200
k_list = list(range(1,kMax+1))

for k in range(1,kMax+1):
    KNNmodel = KNeighborsClassifier(n_neighbors = k)
    KNNmodel.fit(X_train_scaled, y_train)
    Knn_Acc.append(KNNmodel.score(X_test_scaled, y_test))
    
    
plt.plot(k_list, Knn_Acc)
plt.xlabel("k")
plt.ylabel("Set Accuracy")
plt.title("Accuracy of Model on Test Set")
plt.show()

l_np = np.asarray(Knn_Acc)

print(f"Best K for best accuracy is: {l_np.argmax()+1}")'''
#bestK = l_np.argmax()+1
bestK = 189

# Perform regression using best K
'''KNNmodel = KNeighborsClassifier(n_neighbors = bestK)
KNNmodel.fit(X_train_scaled, y_train)

training_accuracy = KNNmodel.score(X_train_scaled, y_train)
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()

predictions = KNNmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''
#model data is 31%, the predictions are even at 31%

##############
#Decision Tree
'''param_grid = {"max_depth": [8, 12, 18, 24, None],
              "min_samples_leaf": range(1, 5),
              "criterion": ["gini", "entropy"]}

DTmodel = DecisionTreeClassifier()

tree_cv = GridSearchCV(DTmodel, param_grid, cv = 5)
tree_cv.fit(X_train_scaled, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# 31%, still pretty bad

### Tuned decision tree
DTmodel = DecisionTreeClassifier(criterion =  'gini' , max_depth =  8, min_samples_leaf = 2)
DTmodel.fit(X_train_scaled, y_train)

training_accuracy = DTmodel.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()

predictions = DTmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))

print(DTmodel.get_depth())'''


##############
# Support Vecotr Machine
# this either takes a very long time, or isn't working!
'''param_grid = {'C': [0.1, 1, 10], 
              'gamma': [0.1, 1, 10],
              'kernel': ['linear', 'rbf']}

SVCmodel = SVC()
svm_cv = GridSearchCV(SVCmodel, param_grid, cv = 5)
svm_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Support Vector Machine classifier Parameters: {}".format(svm_cv.best_params_))
print("Best score is {}".format(svm_cv.best_score_))


SVCmodel = SVC(kernel = 'rbf', C = 0.1, gamma = 1)
SVCmodel.fit(X_train_scaled, y_train)
training_accuracy = SVCmodel.score(X_train_scaled, y_train)
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()

predictions = SVCmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))
'''

##############################
# conclusion:
#   all the models are around 30% accuracy which is poor for a model. 
# This means the data cannot be used to accurately predict body type. This would mostly 
# be due to the fact that the originating data is how the user sees themselves, and is in no
# way empirical data.
##############################
# Question 3
# Predict sex with education level and income??
'''df_copy3 = df
df_copy3[df_copy3.income==-1]= np.nan
temp = df_copy3.income.value_counts(dropna = False)
#print(temp)
selected_features = ['income', 'job', 'sex', 'education']
df_copy3 = df_copy3[selected_features].dropna()

#convert 'job', 'sex' and 'education' into dummy variables
categoricalCols = ['job', 'sex', 'education']
for col in categoricalCols:
    df_copy3 = pd.get_dummies(df_copy3, columns=[col], prefix = [col], drop_first =True)

#print(df_copy3.shape)

#print(df_copy3.head(15))
#Y is the target column, X has the features
X = df_copy3.iloc[:, 1:]
y = df_copy3['income']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 128)

#Pandas Series.ravel() function returns the flattened underlying data as an ndarray(1d array)
y_train = y_train.ravel()
y_test = y_test.ravel()'''


###############
# K Nearest Neighbour
'''accuracies = []
k_list = list(range(1,101))

for k in range(1,101):
    KNNmodel = KNeighborsClassifier(n_neighbors = k)
    KNNmodel.fit(X_train, y_train)
    accuracies.append(KNNmodel.score(X_test, y_test))
    
    
#plt.plot(k_list, accuracies)
#plt.xlabel("k")
#plt.ylabel("Test set Accuracy")
#plt.title("Accuracy of Model on Test Set")
#plt.show()

l_np = np.asarray(accuracies)
bestK = l_np.argmax()+1
print(f"Best K for best accuracy is: {bestK}")


KNNmodel = KNeighborsClassifier(n_neighbors = bestK).fit(X_train, y_train)
KNN_predictions = KNNmodel.predict(X_test)
training_accuracy = KNNmodel.score(X_train, y_train)
print(classification_report(y_test, KNN_predictions))

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''

# model, 36%, output = 38%

#################
# Decision tree classifier
'''param_grid = {"max_depth": [8, 12, 18, 24, None],
              "min_samples_leaf": range(1, 9),
              "criterion": ["gini", "entropy"]}

DTmodel = DecisionTreeClassifier()
tree_cv = GridSearchCV(DTmodel, param_grid, cv = 5)
tree_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

DTmodel = DecisionTreeClassifier(criterion =  'gini' , max_depth =  18, min_samples_leaf = 3)
DTmodel.fit(X_train, y_train)
training_accuracy = DTmodel.score(X_train, y_train)
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()
predictions = DTmodel.predict(X_test)

print(classification_report(y_test, predictions))'''

# output model is 38%, prediction accuracy is 37%

########
# SVC Model
'''SVCmodel = SVC(kernel = 'linear', C = 1)
SVCmodel.fit(X_train, y_train)
training_accuracy = SVCmodel.score(X_train, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()

predictions = SVCmodel.predict(X_test)
print(classification_report(y_test, predictions))'''
# 38% accuracy

##############
# Logistic Regression
'''logReg_model = LogisticRegression(multi_class= "multinomial")
logReg_model.fit(X_train, y_train)
training_accuracy = logReg_model.score(X_train, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
print()

predictions = logReg_model.predict(X_test)
print(classification_report(y_test, predictions))'''

# data accuracy is 37%, prediction is 38%

#######
# all models well under 50% so not a good data source for the hypothesis


###########################
#Question 4
# Predict sex based on age and body type
'''q4data = df
selected_features = ['sex', 'body_type_code', 'height']
q4data = q4data[selected_features].dropna()

#print(q4data.shape)
#print(q4data.head())

#Y is the target column, X has the features
X = q4data.iloc[:, 1:]
y = q4data['sex']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#Pandas Series.ravel() function returns the flattened underlying data as an ndarray(1d array)
y_train = y_train.ravel()
y_test = y_test.ravel()

scaler = StandardScaler()

# standardization 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)

scaler2 = MinMaxScaler()
 
# normalization 
X_train_normalized = scaler2.fit_transform(X_train) 
X_test_normalized = scaler2.fit_transform(X_test)'''

##########
import time
# Perform Multinomial NB
'''NBmodel = MultinomialNB()
#NBmodel = GaussianNB()
start_time = time.time()

NBmodel.fit(X_train_normalized, y_train)
training_accuracy = NBmodel.score(X_train_normalized, y_train)
predictions = NBmodel.predict(X_test_normalized)

print(classification_report(y_test, predictions))
print()

end_time = time.time()
runtime =  end_time - start_time

print(f"The runtime of Multinomial Naive Bayes model is: {round(runtime, 5)} seconds")
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''

#################
# KNN
'''accuracies = []
maxK = 100
k_list = list(range(1,maxK+1))

for k in range(1,maxK+1):
    KNNmodel = KNeighborsClassifier(n_neighbors = k)
    KNNmodel.fit(X_train_scaled, y_train)
    accuracies.append(KNNmodel.score(X_test_scaled, y_test))
    
    
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Test set Accuracy")
plt.title("Accuracy of Model on Test Set")
plt.show()

l_np = np.asarray(accuracies)
bestK = l_np.argmax()+1
print(f"Best K for best accuracy is: {bestK}")

start_time = time.time()
KNNmodel = KNeighborsClassifier(n_neighbors = bestK).fit(X_train_scaled, y_train)
KNN_predictions = KNNmodel.predict(X_test_scaled)
training_accuracy = KNNmodel.score(X_train_scaled, y_train)
print(classification_report(y_test, KNN_predictions))
end_time = time.time()
runtime =  end_time - start_time

print(f"The runtime of K Nearest Neighbor model is: {round(runtime, 5)} seconds")
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''

# bestK was 12, accuracy on training data is 73%, resulting score is 88%

#####
# Decision Tree
'''param_grid = {"max_depth": [8, 12, 18, 20, None],
              "min_samples_leaf": list(range(1, 9)),
              "criterion": ["gini", "entropy"]}

DTmodel = DecisionTreeClassifier()
tree_cv = GridSearchCV(DTmodel, param_grid, cv = 5)
tree_cv.fit(X_train_scaled, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

start_time = time.time()
DTmodel = DecisionTreeClassifier(criterion = 'gini', max_depth = 8, min_samples_leaf = 4)
DTmodel.fit(X_train_scaled, y_train)
training_accuracy = DTmodel.score(X_train_scaled, y_train)
predictions = DTmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))
end_time = time.time()
runtime =  end_time - start_time

print(f"The runtime of Decision Tree model is: {round(runtime, 5)} seconds")
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''

#print(DTmodel.get_depth())
# best score 87.9%
# accuracy of model 89%

############
# SVM
'''start_time = time.time()
SVCmodel = SVC(kernel = 'linear', C = 1)
SVCmodel.fit(X_train_scaled, y_train)
training_accuracy = SVCmodel.score(X_train_scaled, y_train)
predictions = SVCmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))
end_time = time.time()
runtime =  end_time - start_time

print(f"The runtime of SVM model is: {round(runtime, 5)} seconds")
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''
# class report 86%, score on data was 87%

##################
# Logistic Regression
'''start_time = time.time()
logReg_model = LogisticRegression(multi_class= "multinomial")
logReg_model.fit(X_train_scaled, y_train)
training_accuracy = logReg_model.score(X_train_scaled, y_train)
predictions = logReg_model.predict(X_test_scaled)
print(classification_report(y_test, predictions))
end_time = time.time()
runtime =  end_time - start_time

print(f"The runtime of Logistic Regression model is: {round(runtime, 5)} seconds")
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))'''
# class report 86%, accuracy 87%

# All models were quite good at predicting the sex based on the given data


###############################
# Question 5
# Can we predict education level and income? How about word counts?
###############################
'''q5data = df
q5data[q5data.income==-1]= np.nan
selected_features = ['sex', 'education', 'income']
q5data = q5data[selected_features].dropna()

#convert 'education' into dummy variable
q5data = pd.get_dummies(q5data, columns=['education'], prefix = ['education'])

#print(q5data.shape)
#print(q5data.head())

#Y is the target column, X has the features
X = q5data.iloc[:, 1:]
y = q5data['sex']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#Pandas Series.ravel() function returns the flattened underlying data as an ndarray(1d array)
y_train = y_train.ravel()
y_test = y_test.ravel()

#scaler = StandardScaler()

scaler = MinMaxScaler()
 
# normalization
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)'''

########### 
# Multi NB
'''NBmodel = MultinomialNB()
NBmodel.fit(X_train_scaled, y_train)
training_accuracy = NBmodel.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = NBmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''

# model accuracy 73%, data accuracy 71%


################
# KNN
'''accuracies = []
maxK = 100
k_list = list(range(1,maxK+1))

for k in range(1,maxK+1):
    KNNmodel = KNeighborsClassifier(n_neighbors = k)
    KNNmodel.fit(X_train_scaled, y_train)
    accuracies.append(KNNmodel.score(X_test_scaled, y_test))'''

'''plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Test set Accuracy")
plt.title("Accuracy of Model on Test Set")
plt.show()'''

'''l_np = np.asarray(accuracies)
bestK = l_np.argmax()+1
print(f"Best K for best accuracy is: {bestK}")

KNNmodel = KNeighborsClassifier(n_neighbors = bestK)
KNNmodel.fit(X_train_scaled, y_train)
training_accuracy = KNNmodel.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = KNNmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''

# bestK was 29
# mode accuracy 73%, score 72%

################
# Logistic Regression
'''logReg_model = LogisticRegression(multi_class= "multinomial")
logReg_model.fit(X_train_scaled, y_train)
training_accuracy = logReg_model.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = logReg_model.predict(X_test_scaled)

print(classification_report(y_test, predictions))'''
# mode accuracy 73%, score 72%

# do for decision tree, and support vector machine later....much of a muchness though

# All models were around 73% accuracy

#########################################
#Question 6: Predict education level with essay text word counts
# clean the text of links and hashtags

import string
from html.parser import HTMLParser
import re       # regex library

#<.*?>
#r'<[^>]+>'
TAG_RE = re.compile(r'<.*?>')
def remove_tags(dirty_string):
    return re.sub(TAG_RE, '', dirty_string)


def cleanText(dirty_text):
    #Removing URLs and Hashtags
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.\S+', "", dirty_text)
    # remove hashtags
    text = re.sub(r'#', '', text)
    # remove html tags from text
    text = remove_tags(text)
    # removing HTML characters
    #text = HTMLParser().unescape(text)
    #remove punctuation marks from the text
    for c in string.punctuation:
        if c in text:
            text = text.replace(c, "")
    return text

q6data = df

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = q6data[essay_cols].replace(np.nan, '', regex=True)

# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#clean the dirty text from hyperlinks, punctuation and html tags
q6data["all_essays_cleaned_text"] = all_essays.apply(cleanText)

#compute the length of each essay and save them on a new column
q6data["essay_len"] = q6data["all_essays_cleaned_text"].apply(lambda x: len(x))

#count the number of words in each essay
q6data["word_count"] =  q6data["all_essays_cleaned_text"].apply(lambda x: len(x.split())) 

#compute the avrage length of each word
#q6data["avg_word_len_temp"] = q6data['essay_len'] / q6data['word_count']
q6data["avg_word_len"] = q6data.apply(lambda row: 0 if row.word_count==0 else (row.essay_len/row.word_count), axis = 1)
#count the number of "i" or "me" occurances in the essay text of each user
q6data["i_or_me_count"] = q6data["all_essays_cleaned_text"].apply(lambda x: x.split().count('i') + x.split().count('me'))

#create copy of dataframe to do the changes only to a copy of data
df_copy6 = q6data
selected_features = ['education', 'word_count']
df_copy6 = df_copy6[selected_features].dropna()
#print(df_copy6.shape)
#print(df_copy6.head(10))


########## Data extracted, transform and prepare it 
#Y is the response variable, X is the predictor variable
X = df_copy6[['word_count']]
y = df_copy6['education']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

scaler = StandardScaler()
#normalization
#scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)


##################
# Logistic Regression 
'''logReg_model = LogisticRegression(multi_class= "multinomial")
logReg_model.fit(X_train_scaled, y_train)
training_accuracy = logReg_model.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = logReg_model.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''

# 45% acc

################
# SVC
'''SVCmodel = SVC(kernel = 'linear', C = 1)
SVCmodel.fit(X_train_scaled, y_train)
training_accuracy = SVCmodel.score(X_train_scaled, y_train)

print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = SVCmodel.predict(X_test_scaled)
print(classification_report(y_test, predictions))'''

# this one is slow, and has 45% accuracy on both training data and set

####################
# Decision tree
'''DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train_scaled, y_train)
training_accuracy = DTmodel.score(X_train_scaled, y_train)
print("The accuracy of model on training data is: {}%".format(round(training_accuracy, 2) *100))
predictions = DTmodel.predict(X_test_scaled)

print(classification_report(y_test, predictions))'''
# 46% accuracy on model, 44% on data

# Word count doesn't really have a correlation for education level




##################
# Question 7: can we predict income with length of essays and average word length?

#create copy of dataframe to do the changes only to a copy of data
'''df_copy7 = q6data
selected_features = ['income', 'essay_len', 'avg_word_len']
df_copy7[df_copy7.income==-1]= np.nan
df_copy7 = df_copy7[selected_features].dropna()'''

#print(df_copy7.shape)
#print(df_copy7.head(10))

##### Transform the data
#Y is the response variable, X has the predictor variables
'''X = df_copy7[['essay_len', 'avg_word_len']]
y = df_copy7['income']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

scaler = StandardScaler()
#normalization
#scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)'''

##############
# Create linear regression model
# creating a regression model
'''mlr = LinearRegression()

# fitting the model with training data
mlr.fit(X_train_scaled,y_train)

print(f"coeficient for linear regression model is: {mlr.coef_}")
print(f"intercept for our model is: {mlr.intercept_}")

predictions = mlr.predict(X_test_scaled)'''

# model evaluation
'''print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))'''

'''r_squared = mlr.score(X_test_scaled, y_test)

print(f"R2 value of our model is: {r_squared}")
df_copy7.income.describe()'''


# far too much error here, and a very low R2 value makes this mode very inaccurate for the provided data

########################
# Question 8: can we predict age with the frequency of "I" or "me" in essays?
df_copy8 = q6data
selected_features = ['age', 'i_or_me_count']
df_copy8 = df_copy8[selected_features].dropna()

#print(df_copy8.shape)
# output data looks pretty uniform, it's probably not going to work well unless we can normalise the age counts
#print(df_copy8.head(10))

#Y is the response variable, X is predictor variable
X = df_copy8[['i_or_me_count']]
y = df_copy8['age']

#Split the data into training set and test set
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

scaler = StandardScaler()
#normalization
#scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)

'''

#create copy of dataframe to do the changes only to a copy of data
#Y is the response variable, X is predictor variable
X = df_copy8[['i_or_me_count']]
y = df_copy8['age']

#Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

scaler = StandardScaler()
#normalization
#scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)


###############
# Linear regression model
'''start_time = time.time()
# creating a regression model
lr_model = LinearRegression()
# fitting the model with training data
lr_model.fit(X_train_scaled,y_train)
predictions = lr_model.predict(X_test_scaled)
end_time = time.time()

runtime =  end_time - start_time

print(f"coeficient for linear regression model is: {lr_model.coef_}")
print(f"intercept for our model is: {lr_model.intercept_}")

# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"RMSE of linear regression model is: {rmse}")
r_squared = lr_model.score(X_test_scaled, y_test)
print(f"R2 value of our model is: {r_squared}")
print(f"The runtime of linear regression model is: {round(runtime, 5)} seconds")'''

########################
# KNN
'''from sklearn.neighbors import KNeighborsRegressor

gridsearch = GridSearchCV(estimator=KNeighborsRegressor(),
             param_grid={'n_neighbors':list(range(1, 100)),
                         'weights': ['uniform', 'distance']})
gridsearch.fit(X_train_scaled, y_train)

print(gridsearch.best_params_)'''
# best params 93

####
'''start_time = time.time()
knn_model = KNeighborsRegressor(n_neighbors= 96, weights = 'uniform')
knn_model.fit(X_train_scaled, y_train)
predictions = knn_model.predict(X_test_scaled)
end_time = time.time()
runtime =  end_time - start_time

# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE of knn model is: {rmse}")
r_squared = knn_model.score(X_test_scaled, y_test)
print(f"R2 value of our model is: {r_squared}")
print(f"The runtime of KNeighbors Regressor model is: {round(runtime, 5)} seconds")'''

#### TO DO:
#normalise age based on total number of respondants
# normalise sex based on total number and re-run these tests