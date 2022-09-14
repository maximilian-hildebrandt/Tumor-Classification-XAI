from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
import pandas as pd
import numpy as np
from data import get_radiomics_dataset
from sklearn.metrics import accuracy_score

# Load data, concatenate val and train set due to small sample and use of cross validation
train_data, train_labels, val_data, val_labels, test_data, test_labels = get_radiomics_dataset()
train_data_new = pd.concat([train_data, val_data])
train_labels_new = np.append(train_labels, val_labels)

# Preprocess data, e.g., standardize to enable coefficient sorting
features = train_data_new.columns
X = train_data_new.to_numpy()
y = train_labels_new
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# Set hyperparameter
n_folds = 10
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Define and train model
# Max_iter set to 10k to enable convergence, l1 reg. for feature selection, sage solver since faster than liblinear
log = LogisticRegressionCV(Cs=C_values, cv=n_folds, penalty="l1", solver="saga", max_iter=10000, random_state=43)
log.fit(X_scaled, y)

coef_table = pd.DataFrame(list(features)).copy()
coef_table.insert(len(coef_table.columns),"coefs",log.coef_.transpose())
coef_table = coef_table.sort_values(by="coefs", ascending=False)
coef_table.columns = ("feature", "coefs")

# Plot the feature importance in a horizontal bar chart
import matplotlib.pyplot as plt
n_features = 5
Features = coef_table.feature[0:n_features].iloc[::-1]
Importance_Score = coef_table.coefs[0:n_features].iloc[::-1]

plt.barh(Features, Importance_Score)
plt.title('Top 5 Standardized Coefficients  Logistic Regression')
plt.ylabel('Feature')
plt.xlabel('Standardized Coefficients')
plt.savefig('Feature_Importance_Logistic_Regression.png')
plt.show()

# generate predictions
y_pred = log.predict(test_data.iloc[:,:].values)
# score predictions
print("Accuracy on test set:", round(accuracy_score(test_labels, y_pred),4))

