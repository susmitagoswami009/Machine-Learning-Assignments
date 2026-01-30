import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product as cartesian_product

from sklearn.model_selection import train_test_split

def split_dataset(X, y, stratify=None):
  """
  Split the dataset into training, validation, and test sets
  in the ratio of 70:15:15.

  Parameters:
  - X: Features (independent variables)
  - y: Target variable (dependent variable)

  Returns:
  - (Xtrain, ytrain): Training set
  - (Xval, yval): Validation set
  - (Xtest, ytest): Test set
  """
  Xtrain, not_Xtrain, ytrain, not_ytrain = train_test_split(X, y, train_size=0.7, stratify=stratify, random_state=42);
  Xval, Xtest, yval, ytest = train_test_split(not_Xtrain, not_ytrain, train_size=0.5, random_state=42);

  return (Xtrain, ytrain), (Xval, yval), (Xtest, ytest)


# Tests

def test_split_dataset():
  dataset_size = 100
  X = np.random.rand(dataset_size, 2)
  y = np.random.randint(2, size=dataset_size)

  (Xtrain, ytrain), (Xval, yval), (Xtest, ytest) = split_dataset(X, y)

  train_size = len(Xtrain) / dataset_size
  val_size = len(Xval) / dataset_size
  test_size = len(Xtest) / dataset_size

  print(f"Train size: {train_size}")
  print(f"Val size: {val_size}")
  print(f"Test size: {test_size}")

  assert (train_size + val_size + test_size) == 1
  assert 0.68 <= train_size <= 0.72
  assert 0.13 <= val_size <= 0.17
  assert 0.13 <= test_size <= 0.17

test_split_dataset()

def calculate_true_positives(predicted_classes, actual_classes):
  return np.sum((actual_classes == 1) & (predicted_classes == 1))

# Tests
pred_1 = np.array([1, 0, 1, 1])
pred_2 = np.array([0, 0, 0, 0])
actual = np.array([1, 0, 0, 1])

assert calculate_true_positives(pred_1, actual) == 2
assert calculate_true_positives(pred_2, actual) == 0

def calculate_false_positives(predicted_classes, actual_classes):
  return np.sum((actual_classes == 0) & (predicted_classes == 1))

# Tests
pred_1 = np.array([1, 1, 1, 1])
pred_2 = np.array([1, 0, 0, 0])
actual = np.array([1, 0, 0, 1])

assert calculate_false_positives(pred_1, actual) == 2
assert calculate_false_positives(pred_2, actual) == 0

def calculate_true_negatives(predicted_classes, actual_classes):
  return np.sum((actual_classes == 0) & (predicted_classes == 0))

# Tests
pred_1 = np.array([0, 0, 0, 0])
pred_2 = np.array([1, 0, 1, 1])
actual = np.array([0, 1, 0, 1])

assert calculate_true_negatives(pred_1, actual) == 2
assert calculate_true_negatives(pred_2, actual) == 0

def calculate_false_negatives(predicted_classes, actual_classes):
  return np.sum((actual_classes == 1) & (predicted_classes == 0))

# Tests
pred_1 = np.array([0, 0, 0, 0])
pred_2 = np.array([0, 0, 1, 1])
actual = np.array([1, 0, 0, 1])

assert calculate_false_negatives(pred_1, actual) == 2
assert calculate_false_negatives(pred_2, actual) == 1

def evaluate_precision(predicted_classes, actual_classes):
  tp = calculate_true_positives(predicted_classes, actual_classes)
  fp = calculate_false_positives(predicted_classes, actual_classes)

  denom = tp + fp
  if denom == 0:
      return 0.0

  return tp / denom

# Tests
pred_1 = np.array([1, 1, 1, 1])
pred_2 = np.array([1, 0, 0, 0])
actual = np.array([1, 0, 0, 1])

assert evaluate_precision(pred_1, actual) == 0.5
assert evaluate_precision(pred_2, actual) == 1

def evaluate_recall(predicted_classes, actual_classes):
  tp = calculate_true_positives(predicted_classes, actual_classes)
  fn = calculate_false_negatives(predicted_classes, actual_classes)

  denom = tp + fn
  if denom == 0:
      return 0.0

  return tp / denom

# Tests
pred_1 = np.array([1, 1, 1, 0])
pred_2 = np.array([0, 0, 0, 1])
actual = np.array([1, 1, 0, 1])

assert evaluate_recall(pred_1, actual) == (2/3)
assert evaluate_recall(pred_2, actual) == (1/3)

def evaluate_f1_score(predicted_classes, actual_classes):
  precision = evaluate_precision(predicted_classes, actual_classes)
  recall = evaluate_recall(predicted_classes, actual_classes)

  denom = precision + recall
  if denom == 0:
      return 0.0

  return 2 * (precision * recall) / denom

def evaluate_accuracy(predicted_classes, actual_classes):
  np_predicted = np.array(predicted_classes)
  np_actual = np.array(actual_classes)

  accuracy = np.mean(np_actual == np_predicted) * 100

  return np.round(accuracy, 2)

def evaluate_confusion_matrix(predicted_classes, actual_classes):
  tp = calculate_true_positives(predicted_classes, actual_classes)
  fp = calculate_false_positives(predicted_classes, actual_classes)
  tn = calculate_true_negatives(predicted_classes, actual_classes)
  fn = calculate_false_negatives(predicted_classes, actual_classes)

  return np.array([[tp, fp], [fn, tn]])

def generate_classification_report(predicted_classes, actual_classes):
  accuracy = evaluate_accuracy(predicted_classes, actual_classes)
  precision = evaluate_precision(predicted_classes, actual_classes)
  recall = evaluate_recall(predicted_classes, actual_classes)
  f1_score = evaluate_f1_score(predicted_classes, actual_classes)

  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
  }

def print_classification_report(report):
  print(f"Accuracy: {report['accuracy']}")
  print(f"Precision: {report['precision']}")
  print(f"Recall: {report['recall']}")
  print(f"F1 Score: {report['f1_score']}")

def draw_confusion_matrix(confusion_matrix, title):
  fig, ax = plt.subplots()
  ax.imshow(confusion_matrix, cmap='Blues', alpha=0.5)
  ax.set_xticks([0, 1])
  ax.set_xticklabels(['Positive', 'Negative'])
  ax.set_xlabel('True Class')
  ax.xaxis.set_label_position('top')
  ax.xaxis.tick_top()

  ax.set_yticks([0, 1])
  ax.set_yticklabels(['Positive', 'Negative'])
  ax.set_ylabel('Predicted Class')

  entry_labels = [['TP', 'FP'], ['FN', 'TN']]
  for i in range(2):
    for j in range(2):
      text = f'{str(confusion_matrix[i, j])}\n{entry_labels[i][j]}'
      ax.text(j, i, text, ha='center', va='center', color='black')

  fig.suptitle(title, fontsize=16)
  fig.tight_layout()

def convert_to_numpy(*args):
  """
  Convert the given arguments to numpy arrays.

  Parameters:
  - *
  """
  return [np.array(arg) for arg in args]

bcdf = pd.read_csv('DT-BrainCancer.csv')
bcdf = bcdf.drop(columns=['Unnamed: 0'])

bcdf.head()

bcdf.info()

row_containing_nan_diagnosis = bcdf[bcdf['diagnosis'].isna()]

if not row_containing_nan_diagnosis.empty:
  print("Row containing NaN diagnosis:\n", row_containing_nan_diagnosis)
  bcdf = bcdf.drop(row_containing_nan_diagnosis.index).reset_index(drop=True)
  print(bcdf.info())

status_value_counts = bcdf['status'].value_counts()
print(status_value_counts)

fig, ax = plt.subplots()

ax.bar(['Alive', 'Dead'], status_value_counts, color=['green', 'red'])
ax.set_xlabel('Status')
ax.set_ylabel('Patients')
ax.set_title('Patient Survival Status Distribution')

print(f"Class imbalance ratio: " + str(status_value_counts[0]/status_value_counts[1]))

class ZeroRClassifier:

  def __init__(self,):
    self.majority_class = None;
    self.majority_class_proba = None;


  def fit(self, X, y):
    classes, class_counts = np.unique(y, return_counts=True);

    index_of_majority_class = np.argmax(class_counts)

    self.majority_class = classes[index_of_majority_class]
    self.majority_class_proba = class_counts[index_of_majority_class] / len(y)

  def predict(self, X):
    if self.majority_class is None:
      raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")
    return np.full(X.shape[0], self.majority_class)

  def predict_proba(self, X):
    if self.majority_class is None:
      raise RuntimeError("Classifier has not been fitted yet. Call .fit() first.")

    return np.full(X.shape[0], self.majority_class_proba)

X = bcdf.drop(columns=['status'])
y = bcdf['status']

train, val, test = split_dataset(X, y, y)
Xtrain, ytrain = convert_to_numpy(*train)
Xval, yval = convert_to_numpy(*val)
Xtest, ytest = convert_to_numpy(*test)

zeror_classifier = ZeroRClassifier()
zeror_classifier.fit(Xtrain, ytrain)

len(Xtrain), len(Xval), len(Xtest)

print(f"Majority class: {zeror_classifier.majority_class}")

test_report = generate_classification_report(zeror_classifier.predict(Xtest), ytest)
test_acc_zeror = test_report['accuracy']

print_classification_report(test_report)

ytest_flipped = 1 - ytest
ytest_pred_flipped = 1 - zeror_classifier.predict(Xtest)
test_report = generate_classification_report(ytest_pred_flipped, ytest_flipped)

print_classification_report(test_report)

draw_confusion_matrix(evaluate_confusion_matrix(zeror_classifier.predict(Xtest), ytest), 'Confusion Matrix for Brain Cancer Classification\nUsing ZeroR Classifier')

!pip install -q mlxtend

bcdf_oner = bcdf.copy()

gtv_col = bcdf_oner['gtv']
print(f'min gtv: {gtv_col.min()}')
print(f'max gtv: {gtv_col.max()}')

plt.hist(gtv_col, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of GTV Values')
plt.xlabel('GTV')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

bins = 10
labels = [i for i in range(bins)]
bcdf_oner['gtv_binned'] = pd.cut(gtv_col, bins=bins, labels=labels)

X = bcdf_oner.drop(columns=['status', 'gtv'])
y = bcdf_oner['status']

train, val, test = split_dataset(X, y, y)
Xtrain, ytrain = convert_to_numpy(*train)
Xval, yval = convert_to_numpy(*val)
Xtest, ytest = convert_to_numpy(*test)

from mlxtend.classifier import OneRClassifier
oner = OneRClassifier()

oner.fit(Xtrain, ytrain);

test_report = generate_classification_report(oner.predict(Xtest), ytest)
test_acc_oner = test_report['accuracy']

print_classification_report(test_report)

draw_confusion_matrix(evaluate_confusion_matrix(oner.predict(Xtest), ytest), 'Confusion Matrix for Brain Cancer Classification\nUsing OneR Classifier')

bcdf_knn = bcdf.copy()

bcdf_knn.head()

if 'diagnosis_HG glioma' not in bcdf_knn.columns:
  bcdf_knn = pd.get_dummies(bcdf_knn, columns=['diagnosis', 'sex', 'loc'], dtype='int')
  bcdf_knn = bcdf_knn.drop(columns=['sex_Male', 'loc_Supratentorial'])

bcdf_knn.head()

X = bcdf_knn.drop(columns=['status'])
y = bcdf_knn['status']

train, val, test = split_dataset(X, y, y)
Xtrain, ytrain = convert_to_numpy(*train)
Xval, yval = convert_to_numpy(*val)
Xtest, ytest = convert_to_numpy(*test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)
Xtest_scaled = scaler.transform(Xtest)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='hamming')
knn.fit(Xtrain_scaled, ytrain)

train_acc = evaluate_accuracy(knn.predict(Xtrain_scaled), ytrain)
val_acc = evaluate_accuracy(knn.predict(Xval_scaled), yval)

print(f"Training Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")

test_report = generate_classification_report(knn.predict(Xtest_scaled), ytest)
test_acc_knn = test_report['accuracy']

print_classification_report(test_report)

draw_confusion_matrix(evaluate_confusion_matrix(knn.predict(Xtest_scaled), ytest), 'Confusion Matrix for Brain Cancer Classification\nUsing KNN Classifier')

bcdf_svm = bcdf.copy()

if 'diagnosis_HG glioma' not in bcdf_svm.columns:
  bcdf_svm = pd.get_dummies(bcdf_svm, columns=['diagnosis', 'sex', 'loc'], dtype='int')
  bcdf_svm = bcdf_svm.drop(columns=['sex_Male', 'loc_Supratentorial'])

bcdf_svm.head()

X = bcdf_svm.drop(columns=['status'])
y = bcdf_svm['status']

train, val, test = split_dataset(X, y, y)
Xtrain, ytrain = convert_to_numpy(*train)
Xval, yval = convert_to_numpy(*val)
Xtest, ytest = convert_to_numpy(*test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)
Xtest_scaled = scaler.transform(Xtest)

from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=5,
    gamma='scale',
)

svm.fit(Xtrain_scaled, ytrain)

train_acc = evaluate_accuracy(svm.predict(Xtrain_scaled), ytrain)
val_acc = evaluate_accuracy(svm.predict(Xval_scaled), yval)

print(f"Training Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")

test_report = generate_classification_report(svm.predict(Xtest_scaled), ytest)
test_acc_svm = test_report['accuracy']

print_classification_report(test_report)

draw_confusion_matrix(evaluate_confusion_matrix(svm.predict(Xtest_scaled), ytest), 'Confusion Matrix for Brain Cancer Classification\nUsing SVM Classifier')

bcdf_nb = bcdf.copy()

if 'gtv_binned' not in bcdf_nb.columns:
  bins = 10
  labels = [i for i in range(bins)]
  bcdf_nb['gtv_binned'] = pd.cut(bcdf_nb['gtv'], bins=bins, labels=labels)

from sklearn.preprocessing import OrdinalEncoder

X = bcdf_nb.drop(columns=['status', 'gtv'])
y = bcdf_nb['status']

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

X_encoded = pd.DataFrame(X_encoded, columns=X.columns)
X_encoded.head()

train, val, test = split_dataset(X_encoded, y, y)

Xtrain, ytrain = convert_to_numpy(*train)
Xval, yval = convert_to_numpy(*val)
Xtest, ytest = convert_to_numpy(*test)

from sklearn.naive_bayes import CategoricalNB

nb_classifier = CategoricalNB()
nb_classifier.fit(Xtrain, ytrain)

train_acc = evaluate_accuracy(nb_classifier.predict(Xtrain), ytrain)
val_acc = evaluate_accuracy(nb_classifier.predict(Xval), yval)

print(f"Training Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")

test_report = generate_classification_report(nb_classifier.predict(Xtest), ytest)
test_acc_nb = test_report['accuracy']

print_classification_report(test_report)

draw_confusion_matrix(evaluate_confusion_matrix(nb_classifier.predict(Xtest), ytest), 'Confusion Matrix for Brain Cancer Classification\nUsing Naive Bayes Classifier')

model_accuracies = {
    'ZeroR': test_acc_zeror,
    'OneR': test_acc_oner,
    'KNN': test_acc_knn,
    'SVM': test_acc_svm,
    'Naive Bayes': test_acc_nb
}

models = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'purple'])
plt.xlabel('Classifier')
plt.ylabel('Test Accuracy (%)')
plt.title('Comparison of Test Accuracies Across Classifiers')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}', ha='center')

