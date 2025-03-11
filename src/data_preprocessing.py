import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest

def encode_labels(df):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df

def split_data(df):
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def select_features(X_train_scaled, y_train, X_val_scaled, X_test_scaled):
    selector = SelectKBest(mutual_info_classif, k=100)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_val_selected, X_test_selected

def main():
    file_path = '../Data/raw/emotions.csv'
    df = pd.read_csv(file_path)
    df = encode_labels(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    X_train_selected, X_val_selected, X_test_selected = select_features(X_train_scaled, y_train, X_val_scaled, X_test_scaled)
    
    print("Data Preprocessing Complete.")
    np.save('../Data/processed/X_train_selected.npy', X_train_selected)
    np.save('../Data/processed/X_val_selected.npy', X_val_selected)
    np.save('../Data/processed/X_test_selected.npy', X_test_selected)
    np.save('../Data/processed/y_train.npy', y_train)
    np.save('../Data/processed/y_val.npy', y_val)
    np.save('../Data/processed/y_test.npy', y_test)

if __name__ == "__main__":
    main()
