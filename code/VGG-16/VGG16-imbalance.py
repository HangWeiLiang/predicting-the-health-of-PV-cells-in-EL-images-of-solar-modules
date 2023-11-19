from keras.applications import VGG16
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_imbalence import X_train, y_train, X_test, y_test, mono_X_train, mono_y_train, mono_X_test, mono_y_test, poly_X_train, poly_y_train, poly_X_test, poly_y_test
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)


mono_y_train = to_categorical(mono_y_train, num_classes=4)
mono_y_test = to_categorical(mono_y_test, num_classes=4)
mono_X_train = mono_X_train.reshape(-1, 64, 64, 1)
mono_X_test = mono_X_test.reshape(-1, 64, 64, 1)
mono_X_train_rgb = np.repeat(mono_X_train, 3, axis=-1)
mono_X_test_rgb = np.repeat(mono_X_test, 3, axis=-1)


poly_y_train = to_categorical(poly_y_train, num_classes=4)
poly_y_test = to_categorical(poly_y_test, num_classes=4)
poly_X_train = poly_X_train.reshape(-1, 64, 64, 1)
poly_X_test = poly_X_test.reshape(-1, 64, 64, 1)
poly_X_train_rgb = np.repeat(poly_X_train, 3, axis=-1)
poly_X_test_rgb = np.repeat(poly_X_test, 3, axis=-1)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


for layer in base_model.layers:
    layer.trainable = False


model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


batch_size = 32
epochs = 30
history_total = model.fit(X_train_rgb, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_rgb, y_test))
history_mono = model.fit(mono_X_train_rgb, mono_y_train, epochs=epochs, batch_size=batch_size, validation_data=(mono_X_test_rgb, mono_y_test))
history_poly = model.fit(poly_X_train_rgb, poly_y_train, epochs=epochs, batch_size=batch_size, validation_data=(poly_X_test_rgb, poly_y_test))



total_predictions = model.predict(X_test_rgb)
total_all_predictions = np.argmax(total_predictions, axis=1)
total_targets = np.argmax(y_test, axis=1)

# Mono
mono_predictions = model.predict(mono_X_test_rgb)
mono_all_predictions = np.argmax(mono_predictions, axis=1)
mono_targets = np.argmax(mono_y_test, axis=1)

# Poly
poly_predictions = model.predict(poly_X_test_rgb)
poly_all_predictions = np.argmax(poly_predictions, axis=1)
poly_targets = np.argmax(poly_y_test, axis=1)


# Total
loss, accuracy = model.evaluate(X_test_rgb, y_test)
print(f"Total Loss: {loss:.2f}")
print(f"Total Accuracy: {accuracy*100:.2f}%")
classification_total = classification_report(total_targets, total_all_predictions, zero_division=1)
print("Total Classification Report:")
print(classification_total)

# Mono
loss, accuracy = model.evaluate(mono_X_test_rgb, mono_y_test)
print(f"Mono Loss: {loss:.2f}")
print(f"Mono Accuracy: {accuracy*100:.2f}%")
classification_mono = classification_report(mono_targets, mono_all_predictions, zero_division=1)
print("Mono Classification Report:")
print(classification_mono)

# Poly
loss, accuracy = model.evaluate(poly_X_test_rgb, poly_y_test)
print(f"Poly Loss: {loss:.2f}")
print(f"Poly Accuracy: {accuracy*100:.2f}%")
classification_poly = classification_report(poly_targets, poly_all_predictions, zero_division=1)
print("Poly Classification Report:")
print(classification_poly)





def plot_loss_accuracy(history_total, history_mono, history_poly):
    plt.figure(figsize=(18, 12))

    epochs1 = range(1, len(history_total.history['accuracy']) + 1)
    epochs2 = range(1, len(history_mono.history['accuracy']) + 1)
    epochs3 = range(1, len(history_poly.history['accuracy']) + 1)

    plt.subplot(2, 3, 1)
    plt.plot(epochs1, history_total.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs1, history_total.history.get('val_accuracy', []), 'g-', label='Validation Accuracy')
    plt.title('Total Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs2, history_mono.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs2, history_mono.history.get('val_accuracy', []), 'g-', label='Validation Accuracy')
    plt.title('Mono Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs3, history_poly.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs3, history_poly.history.get('val_accuracy', []), 'g-', label='Validation Accuracy')
    plt.title('Poly Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs1, history_total.history['loss'], 'r-', label='Training Loss')
    plt.plot(epochs1, history_total.history.get('val_loss', []), 'y-', label='Validation Loss')
    plt.title('Total Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs2, history_mono.history['loss'], 'r-', label='Training Loss')
    plt.plot(epochs2, history_mono.history.get('val_loss', []), 'y-', label='Validation Loss')
    plt.title('Mono Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(epochs3, history_poly.history['loss'], 'r-', label='Training Loss')
    plt.plot(epochs3, history_poly.history.get('val_loss', []), 'y-', label='Validation Loss')
    plt.title('Poly Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_loss_accuracy(history_total, history_mono, history_poly)


conf_matrix_total = confusion_matrix(total_targets, total_all_predictions)
conf_matrix_mono = confusion_matrix(mono_targets, mono_all_predictions)
conf_matrix_poly = confusion_matrix(poly_targets, poly_all_predictions)

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel('Actual Label')  
    plt.xlabel('Predicted Label')
    plt.show()


plot_confusion_matrix(conf_matrix_total, "Total Dataset Confusion Matrix")
plot_confusion_matrix(conf_matrix_mono, "Mono Dataset Confusion Matrix")
plot_confusion_matrix(conf_matrix_poly, "Poly Dataset Confusion Matrix")

