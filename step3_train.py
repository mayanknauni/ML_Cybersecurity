# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import joblib
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import os     

# import dataset
dataset = pandas.read_csv("master_dataset.csv")

print (dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('label').size())

# split dataset
array = dataset.values
print (array)
X = array[:,0:26]
Y = array[:,26]
validation_size = 0.30
seed = 42
# Split dataset into training set and test set
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train_set, X_test, Y_train_set, Y_test = model_selection.train_test_split(X_train, Y_train, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Evaluating algorithm model

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s Accuracy: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
print("\n CART results on 30% test set \n")
# Create Decision Tree classifer object
cart = DecisionTreeClassifier()
# Train Decision Tree Classifer
cart.fit(X_train_set, Y_train_set)
#saving the model using joblib 
filename = 'finalized_DT_model.sav'
joblib.dump(cart, filename)
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print (result)
#Predict the response for test dataset
predictions_rfc = cart.predict(X_test)
print("\nCART accuracy test: \n")
print(accuracy_score(Y_test, predictions_rfc))
print(confusion_matrix(Y_test, predictions_rfc))
print(classification_report(Y_test, predictions_rfc))

# Make predictions on test dataset
print("\nCART results on final 30% validation \n")
newcart = DecisionTreeClassifier()
newcart.fit(X_train_set, Y_train_set)
newpredictions_rfc = newcart.predict(X_validation)
print("\nCART accuracy validation: \n")
print(accuracy_score(Y_validation, newpredictions_rfc))
print(confusion_matrix(Y_validation, newpredictions_rfc))
print(classification_report(Y_validation, newpredictions_rfc))
df = dataset.reset_index(drop = False)
