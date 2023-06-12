import cv2
import numpy as np
import os


DATA_PATH = os.path.join(
    "C:/Users/princ/Desktop/ActionDetectionforSignLanguage-main/ActionDetectionforSignLanguage-main/CollectedData/")

# In[3]:


data_label = []
for filename in os.listdir(DATA_PATH):
    data_label.append(filename)

print(data_label)

# In[4]:


data_label = np.array(data_label)

# In[5]:


data_label

# # Image labelling
#

# In[6]:


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf

# In[7]:


label_map = {label: num for num, label in enumerate(data_label)}
print(label_map)


# In[8]:


list(label_map.keys())[list(label_map.values()).index(4)]

# In[14]:


sequences, labels = [], []
for action in data_label:
    start = 0
    end = 20
    for sequence in range(70):
        window = []
        for frame_num in range(start, end):
            res = np.load(os.path.join(DATA_PATH, action, "{}.npy".format("frame_" + str(frame_num) + ".jpg")))

            window.append(res)

        # print(action,sequence,np.array(window).shape)
        sequences.append(window)
        labels.append(label_map[action])
        start = start + 20
        end = end + 20

# In[15]:


np.array(labels).shape

# In[16]:


np.array(sequences).shape

# In[17]:


X = np.asarray(sequences).astype('float32')

# In[18]:


X.shape

# In[19]:


y = to_categorical(labels).astype(int)

# In[20]:


y.shape

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# In[22]:


X_train[0].shape

# # Model building
#

# In[23]:


import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential

# In[24]:


model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(20, 1692)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(data_label.shape[0], activation='softmax')
])

# In[25]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[26]:


model.fit(X_train, y_train, epochs=40)

# In[27]:


model.summary()

# In[36]:


model.evaluate(X_test, y_test)


def pred( seq):
    return model.predict(np.expand_dims(seq, axis=0))[0]