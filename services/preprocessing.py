import json
import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
import pickle 

def getting_values_to_drop(file_name):
    file_path = f"artifacts/{file_name}"
    with open(file_path, 'r') as f:
        columns_to_drop = json.load(f)
    return columns_to_drop
        
    
def fill_null_values(df):
    # Load the imputer from the file
    with open('artifacts/fill_missing_values.pkl', 'rb') as f:
        imputer = pickle.load(f)
    new_arr = imputer.transform(df)
    new_df = pd.DataFrame(new_arr, columns=df.columns)
    return new_df


def feature_engineering_df(df):
    df3 = pd.DataFrame()
    new_columns = []
    for col in df.columns:
        if col == 'churn_probability':
            break
        if (('6' in col)) and col not in ['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','churn_probability']:
            idx = df.columns.get_loc(col)
            new_col_name = col.split('6')[0] + "diff"
            new_columns.append(new_col_name)
            df3[new_col_name] = (df.iloc[:,idx] - df.iloc[:,idx+1]) + (df.iloc[:,idx+1] - df.iloc[:,idx+2])
#         df3[new_col_name] = (df2.iloc[:,idx] + df2.iloc[:,idx+1] +  df2.iloc[:,idx+2])
    df3[['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8']] = df[['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8']]
    df3['aon'] = df['aon']
    df3['date_of_last_rech_6'] = pd.to_datetime(df3['date_of_last_rech_6'])
    df3['date_of_last_rech_7'] = pd.to_datetime(df3['date_of_last_rech_7'])
    df3['date_of_last_rech_8'] = pd.to_datetime(df3['date_of_last_rech_8'])
    df3['date_of_last_rech_diff'] = ((df3['date_of_last_rech_7'] - df3['date_of_last_rech_6']) + (df3['date_of_last_rech_8'] - df3['date_of_last_rech_7'])).dt.days
    df3['date_of_last_rech_diff'].fillna(df3['date_of_last_rech_diff'].median(), inplace=True)
    df3.drop(['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8'], axis=1, inplace=True)
    return df3

def scaling_n_transform(df):

    # scaling
    with open('artifacts/scaling_data.pkl', 'rb') as f:
        scaling = pickle.load(f)
    scaled_df = pd.DataFrame(scaling.transform(df), columns=df.columns)

    #transforming
    with open('artifacts/transforming_data.pkl', 'rb') as f:
        transform = pickle.load(f)
    transformed_df = pd.DataFrame(transform.transform(scaled_df), columns=df.columns)

    return transformed_df


def preprocess_df(df):
    values_to_drop = getting_values_to_drop("columnsToDrop.json")
    df1 = df.drop(values_to_drop, axis=1)
    feature_eng_df = feature_engineering_df(df1)
    values_to_drop = getting_values_to_drop("columnsToDrop_stats.json")
    feature_eng_df = feature_eng_df.drop(values_to_drop, axis=1)
    no_null_df = fill_null_values(feature_eng_df)
    scaled_df = scaling_n_transform(no_null_df)
    return scaled_df


    
