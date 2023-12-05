import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
#from tensorflow.keras import regularizers

# Load your features and labels (X, y)
data = pd.read_csv('data2.csv')
X = data.drop(['audio-name', 'class'], axis=1)
y = data['class']

# Convert the target variable to numerical labels (0 for Male, 1 for Female)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.25, random_state=1)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Get the number of features
n_features = X_train.shape[1]

#regularizador L1
#kernel_regularizer= regularizers.l1(0.005)

# Define MLP architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,
                          #kernel_initializer='ones',
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L1(0.01),
                          #activity_regularizer=tf.keras.regularizers.L2(0.01),
                          input_shape=(n_features,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L1(0.01),),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])


#otimizador
opt = keras.optimizers.Adam(learning_rate=0.1)

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

# Predictions
y_train_pred_probs = model.predict(X_train)
y_val_pred_probs = model.predict(X_val)
# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold = 0.5
y_train_pred = (y_train_pred_probs > threshold).astype(int)
y_val_pred = (y_val_pred_probs > threshold).astype(int)

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix - Training Set:")
print(cm_train)

# Confusion Matrix for Validation Set
cm_val = confusion_matrix(y_val, y_val_pred)
print("\nConfusion Matrix - Validation Set:")
print(cm_val)

# Plot Confusion Matrices
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Training Set')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()
