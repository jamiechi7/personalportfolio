# Jamie Chiang
# TAC 259
# HW5
# Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sn
pd.set_option('display.max_columns', None)

# 1. read csv into pandas dataframe
word = pd.read_csv('A_Z_Handwritten_Data.csv').dropna()

# 2. define feature set and target variable
X = word.iloc[:, 1:]
y = word.iloc[:, 0]

# 3. redefine target so it contains corresponding letters
# mapping numbers to letters
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
             7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
             13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
y = y.map(word_dict)

# 4. print shape of feature set and target variable
print('Features:', X.shape)
print('Target:', y.shape)

# 5. show histogram (countplot) of the letters
sn.countplot(x=y, order=sorted(y.unique()), palette = 'husl')
plt.xlabel('label')
plt.ylabel('count')
plt.show()

# 6. partition the data into train and test sets (70/30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2025, stratify= y)

# 7. scale the train and test features (div by 255)
X_train = X_train / 255
X_test = X_test / 255

# 8. create MLPClassifier model with given values for parameters
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=25, alpha = 0.001, learning_rate_init = 0.01, random_state=2025)

# 9. fit to train the model
model.fit(X_train, y_train)

# 10. plot the loss curve
plt.plot(model.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.show()

# 11. display the accuracy of model
print('Accuracy of Model: ' , model.score(X_test, y_test))

# 12. plot the confusion matrix
from sklearn import metrics
y_pred = model.predict(X_test)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# 13. visualize the predicted letter of first row in test dataset as well as the ACTUAL letter.
# Use first row in test set
first_sample = X_test.iloc[[0]]
first_actual = y_test.iloc[0]
first_pred = model.predict(first_sample)[0]
first_img = first_sample.values.reshape(28, 28)
plt.imshow(first_img, cmap='gray')
plt.title('Predicted letter: ' + str(first_pred) + '  Actual letter: ' + str(first_actual))
plt.show()

# 14. display the actual and predicted letter of a misclassified letter
incorrect = y_test.index[(y_test != y_pred)]
wrong_idx = incorrect[0]
wrong_sample = X_test.loc[[wrong_idx]]
wrong_actual = y_test.loc[wrong_idx]
wrong_pred = model.predict(wrong_sample)[0]
wrong_img = wrong_sample.values.reshape(28, 28)
plt.imshow(wrong_img, cmap='gray')
plt.title('Predicted letter: ' + str(wrong_pred) + '  Actual letter: ' + str(wrong_actual))
plt.show()