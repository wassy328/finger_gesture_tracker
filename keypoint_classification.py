#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


# # パス指定

# In[2]:


dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'


# # 分類数設定

# In[3]:


NUM_CLASSES = 60


# # 学習データ読み込み

# In[4]:


X_dataset = np.loadtxt(dataset, delimiter = ',', dtype = 'float32', usecols = list(range(1, (21 * 3) + 1)))


# In[5]:


y_dataset = np.loadtxt(dataset, delimiter = ',', dtype = 'int32', usecols = (0))


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size = 0.75, random_state = RANDOM_SEED)


# # モデル構築

# In[7]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 3, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax')
])


# In[8]:


model.summary()


# In[9]:


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose = 1, save_weights_only = False)
es_callback = tf.keras.callbacks.EarlyStopping(patience = 20, verbose = 1)


# In[10]:


model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


# # モデル訓練

# In[11]:


model.fit(
    X_train,
    y_train,
    epochs = 1000,
    batch_size = 128,
    validation_data = (X_test, y_test),
    callbacks = [cp_callback, es_callback]
)


# In[12]:


val_loss, val_acc = model.evaluate(X_test, y_test, batch_size = 128)


# In[13]:


model = tf.keras.models.load_model(model_save_path)


# In[14]:


predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # 混同行列

# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report = True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels = labels)
    
    df_cmx = pd.DataFrame(cmx_data, index = labels, columns = labels)
 
    fig, ax = plt.subplots(figsize = (7, 6))
    sns.heatmap(df_cmx, annot = True, fmt = 'g' ,square = False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis = 1)

print_confusion_matrix(y_test, y_pred)


# # Tensorflow-Lite用のモデルへ変換

# In[16]:


model.save(model_save_path, include_optimizer = False)


# In[17]:


tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)


# # 推論テスト

# In[18]:


interpreter = tf.lite.Interpreter(model_path = tflite_save_path)
interpreter.allocate_tensors()


# In[19]:


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[20]:


interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# In[21]:


get_ipython().run_cell_magic('time', '', "interpreter.invoke()\ntflite_results = interpreter.get_tensor(output_details[0]['index'])")


# In[22]:


print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

