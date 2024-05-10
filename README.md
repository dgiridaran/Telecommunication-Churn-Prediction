
# Telecom Churn prediction (End to End)

### Problem Statement

In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn.

In this competition, your goal is to build a machine learning model that is able to predict churning customers based on the features provided for their usage.

### Solution

Implemented a ML model with UI using Stremlit to access the model 
and also containerized the app using Docker.

**For Model Explanation and Insights view this ppt** -> https://docs.google.com/presentation/d/1vRTuQq9-FanOm-nICW_3tHrUzO9go-jxCQw2gCHE7kw/edit?usp=sharing

**Notebook** -> https://github.com/dgiridaran/Telecommunication-Churn-Prediction/blob/main/DSE_SEP2023_Batch1.ipynb

### To run app




###### 1. Using Docker
- pull the image from the dockerhub using this command.
```bash
  docker pull giri2742/telecome_streamlit_app
```
- after the image is pulled to the local, use the command to run the app.
```bash
  docker run -d  -p 8501:8501 giri2742/telecome_streamlit_app
```
###### 2. With out Using Docker
- use this command to clone the repository
```bash
  git clone https://github.com/dgiridaran/Telecommunication-Churn-Prediction.git
```
- After the repository is cloned to the local, create a virtual environment using this command.
```bash
  python -m venv venv
```
- Then install requirements.txt file, using this command.
```bash
  pip install -r requirements.txt
```
- After the requirements file installed, you can run the app. command to run the app.
```bash
  streamlit run app.py
```
**Use the test.csv file to test the model in streamlit app**
