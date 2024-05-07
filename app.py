import streamlit as st 
import pandas as pd
from services.preprocessing import preprocess_df
import pickle 
import matplotlib.pyplot as plt

st.title("Telecom Churn Prediction")

st.header("Upload the Dataset")
uploaded_file = st.file_uploader('Upload a file')

# selected_model = st.selectbox(
#     "Select the Model",
#     ["KNN","XGBoost"]
# )

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_ref = df.copy()
    # df_ = df.copy()
    # st.write(df_)
    new_df = preprocess_df(df)
    # if selected_model == "KNN":
    #     file_path = "artifacts/best_model.pkl"
    #     with open(file_path, "rb") as f:
    #         unpickled_best_model = pickle.load(f)
    # elif selected_model == "XGBoost":
    file_path = "artifacts/gradient_boosting.pkl"
    with open(file_path, "rb") as f:
        best_model = pickle.load(f)
    predict = st.button("Predict")
    if predict:
        churn_prediction_values = best_model.predict(new_df)
        churn_prediction_series = pd.Series(churn_prediction_values).value_counts()
        # perc = (churn_prediction_series / churn_prediction_series.sum()) * 100
        plt.figure(figsize=(10,6), dpi=100)
        plt.bar(x=churn_prediction_series.index, height=churn_prediction_series.values)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.figure(figsize=(10,6), dpi=100)
        plt.pie(churn_prediction_series, labels=['no_churn','churn'], autopct='%.2f%%', shadow=True, explode=[0.1, 0.1])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        df_ref['churn_prediction'] = churn_prediction_values
        st.write(df_ref)
    # df_['churn_prediction'] = churn_prediction
    # st.write(df_[df_['churn_prediction'] == 1])

# st.write(
#     """
# ## Explore
# """
# "man in the mirror"
# )

