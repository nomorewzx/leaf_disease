Kaggle Competition: cassava leaf disease detection

### Introduction
This repo contains the code to complete Kaggle Competition: [Cassava Leaf Disease Detection](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview) and consists of below parts:
#### 1. Data Viwer
This is a simple interactive data viewer created using [streamlit](https://www.streamlit.io/)
To view the dataset, you need to 
1. download dataset and unzip to `./cassava-leaf-disease-classification-data`
2. Run `streamlit run data_viewer_streamlit.py` and visit http://localhost:8501 and there you go.

#### 2. Model training and predicting. 
Using ResNet101V2 NN with one dense layer (256 units) as the classification model.

