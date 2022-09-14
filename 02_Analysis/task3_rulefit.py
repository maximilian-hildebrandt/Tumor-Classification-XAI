# Load libraries
import pandas as pd
import numpy as np
from data import get_radiomics_dataset
from rulefit import RuleFit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load data
train_data, train_labels, val_data, val_labels, test_data, test_labels = get_radiomics_dataset()
train_data_new = pd.concat([train_data, val_data])
train_labels_new = np.append(train_labels, val_labels)

# Preprocessing
features = train_data_new.columns # Save names for RuleFit algo
X = train_data_new.to_numpy()
y = train_labels_new

# Set up models and hyperparameters
rfc = RandomForestClassifier(n_estimators=500, max_depth=10)
rf = RuleFit(tree_generator=rfc, max_rules=200, rfmode= 'classify', Cs = (0.005, 0.01, 0.1, 1, 10, 100, 1000), cv=10, random_state=43)

# Fit model
rf.fit(X, y, feature_names=features)

# Show LogisticCV results to determine best C
np.set_printoptions(suppress=True)
print('Best C:', rf.lscv.C_[0])
print('Cs: %s', rf.lscv.Cs_)
print('Grid scores: %f', rf.lscv.scores_)
# Get non-zero rules, print top 20
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("importance", ascending=False)

# Plot the feature importance in a horizontal bar chart
n_features = 5
feature_length = 50
Features = rules.rule[0:n_features].iloc[::-1]
Features_Shortened = []
for f in Features:
    f = str(f)[0:feature_length]
    f = f+("...")
    Features_Shortened.append(f)
Importance_Score = rules.importance[0:n_features].iloc[::-1]

plt.barh(Features_Shortened, Importance_Score, color = ("#19d4e6", '#2596be', '#2596be', '#2596be', '#2596be'))
plt.title('Top 5 Feature Importance RuleFit')
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.savefig('Feature_Importance_Rulesfit.png')
colors = {'rules':'#2596be', 'linear features':'#19d4e6'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc=4)
plt.show()

# generate predictions
y_pred = rf.predict(test_data.iloc[:,:].values) 

# score predictions
print("Accuracy on test set:", round(accuracy_score(test_labels, y_pred),4))