import json
import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler

def getting_values_to_drop(file_name = "columnsToDrop.json"):
    file_path = f"artifacts/{file_name}"
    try:
        with open(file_path, 'r') as f:
            columns_to_drop = json.load(f)
        return columns_to_drop
    except Exception as e:
        print(e)
        return e
    
def fill_null_values(df):
    for col in df.columns:
        if col not in ["date_of_last_rech_6", "date_of_last_rech_7", "date_of_last_rech_8"]:
            df[col].fillna(df[col].median(), inplace=True)

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

    # droping the columns which failed in statistical test
    col_to_drop = ['roam_og_mou_diff', 'roam_ic_mou_diff']
    df = df.drop(col_to_drop, axis=1)

    # scaling
    for col in df.columns:
        if col != 'churn_probability':
            sc = StandardScaler()
            df[[col]] = sc.fit_transform(df[[col]])

    # transformation
    for col in df.columns:
        if col != 'churn_probability':
            transform = PowerTransformer()
            df[[col]] = transform.fit_transform(df[[col]])

    return df


def preprocess_df(df):
    values_to_drop = getting_values_to_drop()
    df1 = df.drop(values_to_drop, axis=1)
    fill_null_values(df1)
    feature_eng_df = feature_engineering_df(df1)
    scaled_df = scaling_n_transform(feature_eng_df)
    return scaled_df


    
