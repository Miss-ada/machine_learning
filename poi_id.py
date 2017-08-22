#!/usr/bin/python

import sys
import pickle
sys.path.append("../final_project")
from tester import dump_classifier_and_data
from tester import test_classifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi','long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##get to know the data
len(data_dict)

count=0
for employee in data_dict:
    for feature, feature_value in data_dict[employee].iteritems():
        print feature
        count+=1
    break

print count

##check the allocation between the POI and non-POI
import pandas
import seaborn as sns

data_fm=pandas.DataFrame.from_dict(data_dict, orient='columns', dtype=None)
data_fm=data_fm.T

poi_data_frame=data_fm.loc[data_fm['poi'] ==True]
nonpoi_data_frame=data_fm.loc[data_fm['poi'] ==False]

usefulpoi=poi_data_frame.drop(poi_data_frame.columns[[3, 4, 6,7,8,9,13,15,17,18]], axis=1)
usefulnonpoi=nonpoi_data_frame.drop(nonpoi_data_frame.columns[[3, 4, 6,7,8,9,13,15,17,18]], axis=1)
poi_boxplot=sns.boxplot(usefulpoi)
poi_boxplot.set(ylim=(0, 10**8))

poi_number=len(poi_data_frame)

nonpoi_boxplot=sns.boxplot(usefulnonpoi)
nonpoi_boxplot.set(ylim=(0, 10**8))



### Task 2: Remove outliers


for employee in data_dict:
    print employee

data_dict["TOTAL"]
data_dict["THE TRAVEL AGENCY IN THE PARK"]
data_dict.pop("TOTAL", None)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)
data_dict.pop("LOCKHART EUGENE E", None)

### Task 3: Create new feature(s)

for employee in data_dict:
    if(data_dict[employee]['to_messages'] not in ['NaN', 0] and data_dict[employee]['from_this_person_to_poi'] not in ['Nan',0]):
        data_dict[employee]['to_poi_percentage']=float(data_dict[employee]['from_this_person_to_poi'])/(data_dict[employee]['to_messages'])
    else:
        data_dict[employee]['to_poi_percentage']=0

    if(data_dict[employee]['from_messages'] not in ['NaN', 0] and data_dict[employee]['from_poi_to_this_person'] not in ['Nan',0]):
        data_dict[employee]['from_poi_percentage']=float(data_dict[employee]['from_poi_to_this_person'])/(data_dict[employee]['from_messages'])
    else:
        data_dict[employee]['from_poi_percentage']=0


features_list+=['to_poi_percentage', 'from_poi_percentage']
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# from sklearn import cross_validation
# features_train, features_test, labels_train, labels_test=cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
##better split
# from sklearn.cross_validation import StratifiedShuffleSplit
# StratifiedShuffleSplit(y,n_iter=10, test_size=0.5, random_state=0)
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


##use decision tree

from sklearn import tree

updated_features_list=['poi']
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
importance=clf.feature_importances_
for i, feature_importance in enumerate(clf.feature_importances_):
    print features_list[i+1], feature_importance
    if feature_importance !=0.0:
        updated_features_list.append(features_list[i])

# ### Extract features and labels from dataset with updated_features_list.
# data = featureFormat(my_dataset, updated_features_list, sort_keys=True)
# labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
#                                                                                                  test_size=0.3,
#                                                                                                  random_state=42)



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
scaler.fit_transform(features)

pca = PCA()
skb = SelectKBest()
svr = svm.SVC()
nb = GaussianNB()


#features_selected=[features_list[i+1] for i in skb.get_support(indices=True)]

## use pipeline to pass the result of scaling to selector and pass the selected features to PCA and then to GaussianNB()

pipeline = Pipeline(steps=[('scaling',scaler),("SKB", skb),("PCA", pca), ("NB", nb)])


parameters = {"PCA__n_components": [2, 4, 6, 8],
              "PCA__whiten": [True],
              "SKB__k": [12]}

# SKB_k turned within  8, 10, 12, 14, 16, 18 and 12 is the best. the automatic selected is 10.

# 'SVR__kernel':('linear', 'rbf'), 'SVR__C':[1, 10, 100, 1000]}


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

gs=GridSearchCV(pipeline,
                param_grid=parameters,
                scoring="f1",
                cv=sss,
                    error_score=0)
gs.fit(features, labels)
labels_predictions=gs.predict(features_test)

clf=gs.best_estimator_

##find the parameter in SKB
skb_step = gs.best_estimator_.named_steps['SKB']
features_selected=[features_list[i+1] for i in skb_step.get_support(indices=True)]

clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
clf_score=gs.score(features_test, labels_test)


test_classifier(clf, my_dataset, features_list, folds = 1000)

# import cPickle
# f = file('my_dataset.pkl', 'wb')
# cPickle.dump(my_dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
# f.close()
#
# fd = file('my_feature_list.pkl', 'wb')
# cPickle.dump(features_list, fd, protocol=cPickle.HIGHEST_PROTOCOL)
# fd.close()
#
# fc=file('my_classifier.pkl','wb')
# cPickle.dump(clf, fc, protocol=cPickle.HIGHEST_PROTOCOL)
# fc.close()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)