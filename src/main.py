import numpy as np
import pandas as pd
import os
from data_preprocessing import preprocess_data

file_path = '../Data/raw/emotions.csv'
(X_train, X_val, X_test, y_train, y_val, y_test, X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df) = preprocess_data(file_path)

npy_dir = '../Data/processed/npy'
csv_dir = '../Data/processed/csv'
os.makedirs(npy_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

np.save(os.path.join(npy_dir, 'X_train.npy'), X_train)
np.save(os.path.join(npy_dir, 'X_val.npy'), X_val)
np.save(os.path.join(npy_dir, 'X_test.npy'), X_test)
np.save(os.path.join(npy_dir, 'y_train.npy'), y_train)
np.save(os.path.join(npy_dir, 'y_val.npy'), y_val)
np.save(os.path.join(npy_dir, 'y_test.npy'), y_test)

X_train_df.to_csv(os.path.join(csv_dir, 'X_train.csv'), index=False)
X_val_df.to_csv(os.path.join(csv_dir, 'X_val.csv'), index=False)
X_test_df.to_csv(os.path.join(csv_dir, 'X_test.csv'), index=False)
y_train_df.to_csv(os.path.join(csv_dir, 'y_train.csv'), index=False)
y_val_df.to_csv(os.path.join(csv_dir, 'y_val.csv'), index=False)
y_test_df.to_csv(os.path.join(csv_dir, 'y_test.csv'), index=False)