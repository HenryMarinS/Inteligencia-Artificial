#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import streamlit as st
from PIL import Image
model = tf.keras.models.load_model('modeloCabHumV3.hdf5')


# In[10]:


st.write("""
         # Predicción entre caballo :horse: y persona :sunglasses:
         """
         )

image1 = Image.open('man.png')

st.image(image1, caption='Man')


st.write("Alguna vez ha visto alguna vez una foto y ha pensado: ¿es eso un caballo o una persona?")
st.write("No se preocupe es una duda muy común. Por resolver sus dudad suba una foto.")
         
file = st.file_uploader("Por favor, suba su imagen aquí.", type=["jpg", "png"])


# In[11]:


import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Por favor, suba su imagen.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("## Tiene cuatro patas y corre, es un *caballo*. :horse:")
    else:
        st.write("## Parece ser una *persona*. :sunglasses:")
    
    st.text("Probabilidad (0: Caballo, 1: Humano)")
    st.write(prediction)


# In[12]:


st.write("Si desea poner el modelo a prueba puede usar la siguiente imagen:")
image2 = Image.open('a.png')

st.image(image2, caption='¿Humano?')


# In[ ]:





# In[ ]:




