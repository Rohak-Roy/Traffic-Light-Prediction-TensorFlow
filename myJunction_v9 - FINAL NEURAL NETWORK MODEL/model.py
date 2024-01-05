import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# PLEASE NOTE ALL THIS CODE WAS RUN ON GOOGLE COLAB NOTEBOOKS BECAUSE OF THEIR ACCESS TO CLOUD GPUs.

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

dataset_path = '/content/gdrive/MyDrive/data_CO2Emissions_cubed'

def compare_predictions(y_preds, y_test):
  y_preds = tf.round(y_preds.squeeze())
  y_test = np.array(y_test)
  counter = 0

  for i in range(len(y_preds)):
    print(f'Predicted: {y_preds[i]}, Truth: {y_test[i]}\n')
    if y_preds[i] == y_test[i]:
      counter += 1

  print(f'Out of {len(y_test)} data points, {counter} of them are correct.')

df = pd.read_csv(dataset_path)
df.head()

X = df.drop("Traffic Phase", axis=1)
y = df["Traffic Phase"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                y_test,
                                                test_size=0.5,
                                                random_state=42)

mean = []
std = []
for col in X_train.columns:
  mean.append(X_train[col].mean())
  std.append(X_train[col].std())

train_mean = np.array(mean)
train_std = np.array(std)

import csv
header = ["Number Of Vehicles Stopped - Horizontal",	"Total Waiting Time - Horizontal",	"Carbon Emissions Released - Horizontal", "Carbon Emissions Released Squared - Horizontal", "Carbon Emissions Released Cubed - Horizontal",	"Number of Vehicles Predicted at Traffic Light - Horizontal",	"Number of Cars - Horizontal",	"Number of Buses - Horizontal",	"Number of Trucks - Horizontal",	"Number of Motorcycles - Horizontal",	"Number of Bicycles - Horizontal",	"Number Of Vehicles Stopped - Vertical",	"Total Waiting Time - Vertical",	"Carbon Emissions Released - Vertical", "Carbon Emissions Released Squared - Vertical", "Carbon Emissions Released Cubed - Vertical",	"Number of Vehicles Predicted at Traffic Light - Vertical",	"Number of Cars - Vertical",	"Number of Buses - Vertical",	"Number of Trucks - Vertical",	"Number of Motorcycles - Vertical",	"Number of Bicycles - Vertical", "Time Since Signal Change"]
with open('/content/mean_and_std.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  writer.writerow(train_mean)
  writer.writerow(train_std)

X_train_normal = (X_train - train_mean) / train_std
X_val_normal = (X_val - train_mean) / train_std
X_test_normal = (X_test - train_mean) / train_std

def print_mean_and_std(df):
  mean = []
  std = []
  for col in df.columns:
    mean.append(df[col].mean())
    std.append(df[col].std())

  mean = np.array(mean)
  std = np.array(std)

  print(f'mean = {mean}, std = {std}')

X_train_normal.shape, X_test_normal.shape, X_val_normal.shape

X_train_normal_np = X_train_normal.to_numpy()
X_val_normal_np = X_val_normal.to_numpy()
X_test_normal_np = X_test_normal.to_numpy()

X_train_normal_np.shape, X_val_normal_np.shape, X_test_normal_np.shape

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1000, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

history = model.fit(X_train_normal_np, y_train, epochs=100, callbacks=[lr_scheduler])

plt.figure(figsize=(10, 7))
pd.DataFrame(history.history).plot()
plt.xlabel('Epochs')
plt.title("Model Training Curves")

lrs = 1e-4 * (10 ** (np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");

tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1000, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model_2.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(0.0002),
                metrics=['accuracy'])

filepath = 'my_best_model_v7.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')

history_2 = model_2.fit(X_train_normal_np, y_train, epochs=500, validation_data=(X_val_normal_np, y_val), callbacks=[checkpoint])

pd.DataFrame(history_2.history).plot()

plt.plot(np.arange(0, 500), history_2.history['loss'], label='Training Loss')
plt.plot(np.arange(0, 500), history_2.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.plot(np.arange(0, 500), history_2.history['val_loss'], label='Val Loss')
plt.plot(np.arange(0, 500), history_2.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.legend()

best_model = keras.models.load_model('/content/my_best_model_v7.hdf5')

plot_model(best_model, show_shapes=True)

best_model.evaluate(X_test_normal_np, y_test)

y_preds = best_model.predict(X_test_normal_np)
compare_predictions(y_preds, y_test)

import itertools
from sklearn.metrics import confusion_matrix

y_preds_formatted = tf.round(y_preds.squeeze())
y_test_formatted =  np.array(y_test)

cm = confusion_matrix(y_test_formatted, y_preds_formatted)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize=(7, 7))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

labels = np.arange(cm.shape[0])

ax.set(title="Confusion Matrix",
       xlabel="Predicted label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(20)

threshold = (cm.max() + cm.min()) / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
           horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=15)

