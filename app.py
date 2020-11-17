import streamlit as st
import pandas as pd
import joblib
from PIL import Image

image = Image.open('iris.png')
st.image(image, caption='',use_column_width=True)

#Loading Our final trained Knn model 
model= open("Knn_Classifier.pkl", "rb")
knn_clf=joblib.load(model)


st.title("CLASSIFICATORE IRIS")

#Loading images

setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')

st.sidebar.title("Features")

#Intializing
parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']

values=[]

#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
	
	values= st.sidebar.slider(label=parameter, 
                               key=parameter,
                               value=float(parameter_df),
                               min_value=0.0, 
                               max_value=8.0, 
                               step=0.1)
 
	parameter_input_values.append(values)
	
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)

if st.button("Risultato Classificazione"):
	prediction = knn_clf.predict(input_variables)
	st.image(setosa) if prediction == 0 else \
    st.image(versicolor) if prediction == 1 else \
    st.image(virginica)
	st.write('La classe di appartenenza è SETOSA') if prediction == 0 else \
    st.write('La classe di appartenenza è VERSICOLOR')  if prediction == 1 else \
    st.write('La classe di appartenenza è VIRGINICA')