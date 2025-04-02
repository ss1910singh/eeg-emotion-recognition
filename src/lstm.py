import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def run_experiment():
    X_train = np.load('../data/processed/npy/X_train.npy')
    X_test = np.load('../data/processed/npy/X_test.npy')
    X_val = np.load('../data/processed/npy/X_val.npy')
    y_train = np.load('../data/processed/npy/y_train.npy')
    y_test = np.load('../data/processed/npy/y_test.npy')
    y_val = np.load('../data/processed/npy/y_val.npy')

    X_train_reshaped = np.expand_dims(X_train, axis=-1)
    X_val_reshaped = np.expand_dims(X_val, axis=-1)
    X_test_reshaped = np.expand_dims(X_test, axis=-1)

    num_classes = len(np.unique(y_train))
    y_train_oh = np.eye(num_classes)[y_train]
    y_val_oh = np.eye(num_classes)[y_val]
    y_test_oh = np.eye(num_classes)[y_test]

    input_layer = Input(shape=(X_train_reshaped.shape[1], 1))
    lstm_layer = LSTM(250)(input_layer)
    flatten_layer = Flatten()(lstm_layer)
    output_layer = Dense(num_classes, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=10),
        ModelCheckpoint('best_model_k200.h5', save_best_only=True)
    ]

    history = model.fit(
        x=X_train_reshaped,
        y=y_train_oh,
        validation_data=(X_val_reshaped, y_val_oh),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_pred_probs = model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_true_classes = np.argmax(y_test_oh, axis=1)

    metrics_results = {
        'test_accuracy': accuracy_score(y_test_true_classes, y_pred_classes),
        'precision': precision_score(y_test_true_classes, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test_true_classes, y_pred_classes, average='weighted'),
        'f1_score': f1_score(y_test_true_classes, y_pred_classes, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test_true_classes, y_pred_classes).tolist(),
        'classification_report': classification_report(y_test_true_classes, y_pred_classes),
    }

    print(metrics_results)
    cmatrix = confusion_matrix(y_test_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Epochs vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Epochs vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_true_classes, y_pred_classes, alpha=0.5)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Scatter Plot of True vs Predicted Labels')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(y_test_true_classes, bins=num_classes, alpha=0.5, label='True Labels')
    plt.hist(y_pred_classes, bins=num_classes, alpha=0.5, label='Predicted Labels')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Histogram of True and Predicted Labels')
    plt.legend()
    plt.show()

run_experiment()