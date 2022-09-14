#import the libraries
from data import get_radiomics_dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pickle
import matplotlib.pyplot as plt
import argparse
# Parse information from the user
parser = argparse.ArgumentParser()
parser.add_argument("-n","--number_iters",type = int,  \
    help="The number of random forests from the RandomSearchCV.", default=100)
parser.add_argument('-c',"--crossvalidation", type = int, \
        help = "The k of k-CV to be used.", default= 10)
args = parser.parse_args()

#The dataset
train_data, train_labels, val_data, val_labels, test_data, test_labels = get_radiomics_dataset()
train_data_new = pd.concat([train_data, val_data])
train_labels_new = np.append(train_labels, val_labels) # Since the amount of data is low, we combine it and then use CV

#Different hyperparameters for RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 21)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
rf = RandomForestClassifier() #Base classifier initialized
#The grid is created
random_grid = {'n_estimators': n_estimators, 
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,}
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,\
     n_iter = args.number_iters, cv = args.crossvalidation, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(train_data_new, train_labels_new)#Fitting the rf

# pd.DataFrame(rf_random.cv_results_).sort_values(by='rank_test_score')

#Obtain the best random forest
rf_best = rf_random.best_estimator_
rf_best.fit(train_data_new, train_labels_new)
importances = rf_best.feature_importances_ #Most important features of the rf
sorted_indices = np.argsort(importances)[::-1]

#Plot the most important features
plt.title('Feature Importance')
plt.bar(range(train_data_new.shape[1]), importances[sorted_indices], align='center')
plt.savefig('Feature_Importance_task1.png')
plt.show()

#The score on test set
print(f"The test score of the best random forest is {rf_best.score(test_data, test_labels)}")
#Saving the model for future use
with open("model_weights/Task1_best.pkl","wb") as f:
    pickle.dump(rf_best, f)
