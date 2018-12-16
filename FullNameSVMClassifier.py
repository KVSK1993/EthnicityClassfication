import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Reading the Data
sdf = pd.read_csv(r'C:\Users\karanvisingh\Downloads\Karan\First Assignment\Data\wiki_name_race.csv')
#removed null entries
sdf.dropna(subset=['name_first', 'name_last'], inplace=True) 
# first letter capital of first name
sdf['name_first'] = sdf.name_first.str.title() 
# first letter capital of last name
sdf['name_last'] = sdf.name_last.str.title() 

#to see distribution of labels
sdf.groupby('race')['name_first'].count()

#creating full name as  new column
sdf['name_full'] = sdf['name_first'] + ' ' + sdf['name_last']
sdf.sample(frac=1)

#Assigning Category Codes to the classes
y = np.array(sdf.race.astype('category').cat.codes)

#Splitting the data into Test and Train
X_train,  X_test, y_train, y_test = train_test_split(sdf['name_full'], y, test_size=0.2, random_state=21, stratify=y)
#Learning the parameters using HashingVectorizer
vect = HashingVectorizer(analyzer='char', n_features = 325000, ngram_range=(2,4),lowercase=False).fit(X_train)# n_features = 325000,min_df=30, max_df=0.3,max_features = 9000,# ,stop_words = stopWords, sublinear_tf=True,norm = 'l2')#,#2,4 giving best accuracy right now
#Learning the parameters using TfidfVectorizer
#vect = TfidfVectorizer(analyzer='char',min_df=30, max_df=0.3,norm = 'l2',ngram_range=(2,4),lowercase=False).fit(X_train)
#Transforming Test Data
X_train_transform = vect.transform(X_train)

#Transformming Test Data
X_test_transform = vect.transform(X_test)


#Model Fitting
model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train_transform,y_train)
#Model Predictions on Test Data
svcPredictions = model.predict(X_test_transform)

svcAccuracy = accuracy_score(svcPredictions, y_test)
print("SVM Accuracy using HashingVectorizer:", svcAccuracy)

#Plotting Confusion matrix

conf_mat = confusion_matrix(y_test, svcPredictions)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=sdf.race.astype('category').cat.categories, yticklabels=sdf.race.astype('category').cat.categories)
plt.ylabel('Actual HashingVectorizer SVC')
plt.xlabel('Predicted HashingVectorizer SVC')
plt.show()

#Printing Classification Report
print(classification_report(y_test, svcPredictions, target_names=sdf.race.astype('category').cat.categories))