from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Dense, Add, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from data_pre import X_train, y_train, X_test, y_test, mono_X_train, mono_y_train, mono_X_test, mono_y_test, poly_X_train, poly_y_train, poly_X_test, poly_y_test



def create_innovative_cnn_with_dropout(input_shape, num_classes, dropout_rate=0.5):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    shortcut = x
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([shortcut, x])

    x = GlobalAveragePooling2D()(x)

    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(dropout_rate)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test, input_shape, num_classes, dropout_rate=0.5, batch_size=32, epochs=5):

    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    model = create_innovative_cnn_with_dropout(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_encoded, axis=1)

    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print('Test accuracy:', test_accuracy)
    print('F1 Score:', f1)


    return model, history, test_accuracy, conf_matrix


model_total, history_total, test_acc_total, conf_matrix_total = train_and_evaluate_model(X_train, y_train, X_test, y_test, input_shape=(64, 64, 1), num_classes=10, dropout_rate=0.5, batch_size=32, epochs=5)
model_poly, history_poly, test_acc_poly, conf_matrix_poly = train_and_evaluate_model(poly_X_train, poly_y_train, poly_X_test, poly_y_test, input_shape=(64, 64, 1), num_classes=10, dropout_rate=0.5, batch_size=32, epochs=5)
model_mono, history_mono, test_acc_mono, conf_matrix_mono = train_and_evaluate_model(mono_X_train, mono_y_train, mono_X_test, mono_y_test, input_shape=(64, 64, 1), num_classes=10, dropout_rate=0.5, batch_size=32, epochs=5)

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