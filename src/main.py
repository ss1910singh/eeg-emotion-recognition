import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data

file_path = '../Data/raw/emotions.csv'
(X_train, X_val, X_test, y_train, y_val, y_test, 
 X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df) = preprocess_data(file_path)


np.save('../Data/processed/npy/X_train.npy', X_train)
np.save('../Data/processed/npy/X_val.npy', X_val)
np.save('../Data/processed/npy/X_test.npy', X_test)
np.save('../Data/processed/npy/y_train.npy', y_train)
np.save('../Data/processed/npy/y_val.npy', y_val)
np.save('../Data/processed/npy/y_test.npy', y_test)

X_train_df.to_csv('../Data/processed/csv/X_train.csv', index=False)
X_val_df.to_csv('../Data/processed/csv/X_val.csv', index=False)
X_test_df.to_csv('../Data/processed/csv/X_test.csv', index=False)
y_train_df.to_csv('../Data/processed/csv/y_train.csv', index=False)
y_val_df.to_csv('../Data/processed/csv/y_val.csv', index=False)
y_test_df.to_csv('../Data/processed/csv/y_test.csv', index=False)


print("done!!!!!!!!!!!!!!!!")
