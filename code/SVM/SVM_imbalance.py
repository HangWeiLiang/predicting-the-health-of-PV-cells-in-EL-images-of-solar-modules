from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
from data_imbalence import X_train, X_test, y_train, y_test, types_train, types_test
from data_imbalence import mono_X_train, mono_X_test, poly_X_train, poly_X_test, mono_y_train, mono_y_test, poly_y_train, poly_y_test

from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


X_train_mono_flat = mono_X_train.reshape((mono_X_train.shape[0], -1))
X_test_mono_flat = mono_X_test.reshape((mono_X_test.shape[0], -1))

X_train_poly_flat = poly_X_train.reshape((poly_X_train.shape[0], -1))
X_test_poly_flat = poly_X_test.reshape((poly_X_test.shape[0], -1))

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf).fit(X_train_mono_flat, mono_y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf).fit(X_train_poly_flat, poly_y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf).fit(X_train_flat, y_train)

X_mono_train_selected = selector.transform(X_train_mono_flat)
X_mono_test_selected = selector.transform(X_test_mono_flat)

X_poly_train_selected = selector.transform(X_train_poly_flat)
X_poly_test_selected = selector.transform(X_test_poly_flat)

X_train_selected = selector.transform(X_train_flat)
X_test_selected = selector.transform(X_test_flat)

svm_mono = SVC(kernel='rbf')
svm_poly = SVC(kernel='rbf')

svm_total = SVC(kernel='rbf')

svm_mono.fit(X_train_mono_flat, mono_y_train)
svm_poly.fit(X_train_poly_flat, poly_y_train)

svm_total.fit(X_train_flat, y_train)


y_pred_mono = svm_mono.predict(X_test_mono_flat)
y_pred_poly = svm_poly.predict(X_test_poly_flat)

y_pred = svm_total.predict(X_test_flat)

report_mono = classification_report(mono_y_test, y_pred_mono, output_dict=True)
print("Mono Model Evaluation:")
print(report_mono)

f1_mono = report_mono['weighted avg']['f1-score']
precision_mono = report_mono['weighted avg']['precision']
recall_mono = report_mono['weighted avg']['recall']

print("Mono Model - F1 Score: {:.2f}".format(f1_mono))
print("Mono Model - Precision: {:.2f}".format(precision_mono))
print("Mono Model - Recall: {:.2f}".format(recall_mono))
print("Accuracy:", accuracy_score(mono_y_test, y_pred_mono))

report_poly = classification_report(poly_y_test, y_pred_poly, output_dict=True)
print("poly Model Evaluation:")
print(report_poly)

f1_poly = report_poly['weighted avg']['f1-score']
precision_poly = report_poly['weighted avg']['precision']
recall_poly = report_poly['weighted avg']['recall']

print("poly Model - F1 Score: {:.2f}".format(f1_poly))
print("poly Model - Precision: {:.2f}".format(precision_poly))
print("poly Model - Recall: {:.2f}".format(recall_poly))

print("Accuracy:", accuracy_score(poly_y_test, y_pred_poly))

report_total = classification_report(y_test, y_pred, output_dict=True)
print("poly Model Evaluation:")
print(report_total)

f1_total = report_total['weighted avg']['f1-score']
precision_total = report_total['weighted avg']['precision']
recall_total = report_total['weighted avg']['recall']

print("total Model - F1 Score: {:.2f}".format(f1_total))
print("total Model - Precision: {:.2f}".format(precision_total))
print("total Model - Recall: {:.2f}".format(recall_total))
print("Accuracy:", accuracy_score(y_test, y_pred))

conf_matrix_mono = confusion_matrix(mono_y_test, y_pred_mono)
print("Confusion Matrix for Mono Model:")
print(conf_matrix_mono)

conf_matrix_poly = confusion_matrix(poly_y_test, y_pred_poly)
print("\nConfusion Matrix for Poly Model:")
print(conf_matrix_poly)

conf_matrix_total = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix for Total Model:")
print(conf_matrix_total)


df_conf_matrix_mono = pd.DataFrame(conf_matrix_mono)
df_conf_matrix_poly = pd.DataFrame(conf_matrix_poly)
df_conf_matrix_total = pd.DataFrame(conf_matrix_total)

plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
sns.heatmap(df_conf_matrix_mono, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.title("Mono Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
sns.heatmap(df_conf_matrix_poly, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.title("Poly Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
sns.heatmap(df_conf_matrix_total, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.title("Total Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()