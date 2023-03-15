import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import os
from IPython.display import display
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

print(os.listdir("data")) 

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

filenames = os.listdir("data/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# inspect raw data
#display(df)

# plot class sizes
#df['category'].value_counts().plot.bar()
#df['category'].plot(kind='bar')
#plt.show()

sample = random.choice(filenames)
image = load_img("data/train/"+sample)
plt.imshow(image)
plt.show()