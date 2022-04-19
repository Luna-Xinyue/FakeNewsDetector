#!/usr/bin/env python
# coding: utf-8

# In[9]:


import keras
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from unicodedata import normalize as unicodeNormalize


# In[3]:



df = pd.read_csv('./fake_or_real_news.csv')
df['merged_text'] = df['title'] + '. ' + df['text']
df.loc[df['label'] == 'REAL', 'target'] = 0
df.loc[df['label'] == 'FAKE', 'target'] = 1

df_pick = df[['merged_text', 'target']]
df.head()


# 1. Load data as train and test data.
#   a. add Title to text and create final text, label ordered pair.
#   b. Split 80-20 into train and test datasets.

# In[4]:


training_data = df_pick.sample(frac=0.8, random_state=25)
testing_data = df_pick.drop(training_data.index)

print(training_data.shape)
print(testing_data.shape)


# In[18]:


training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(training_data['merged_text'].values, tf.string),
            tf.cast(training_data['target'].values, tf.int32)
        )
    )
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(testing_data['merged_text'].values, tf.string),
            tf.cast(testing_data['target'].values, tf.int32)
        )
    )
)
BUFFER_SIZE = 10000
BATCH_SIZE = 64
training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# In[16]:


for example, label in training_dataset.take(1):
  print('texts: ', example.numpy()[:1])
  print('labels: ', label.numpy()[:1])


# In[19]:


VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(training_dataset.map(lambda text, label: text))


# In[20]:


vocab = np.array(encoder.get_vocabulary())
vocab[:20]


# In[21]:


encoded_example = encoder(example)[:3].numpy()
encoded_example


# In[22]:


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[27]:


sample_text = ('''Poll: 71 percent of Dems think Clinton should keep running even if indicted. A strong majority of Democratic voters think Hillary
               Clinton should keep running for president even if she is charged with
               a felony in connection with her private email use while secretary of state, according to a new poll.''')
predictions = model.predict(np.array([sample_text]))
print(predictions[0])


# In[28]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[29]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(training_dataset, epochs=20, callbacks = [callback],
                    validation_data=test_dataset)


# In[30]:


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


# In[31]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
