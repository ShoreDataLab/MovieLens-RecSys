#%%
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import load_npz
import numpy as np

# Step 1: Load the user-movie sparse matrix
user_movie_sparse = load_npz("../data/processed/user_movie_sparse_matrix.npz")

# Step 2: Apply Truncated SVD to reduce the dimensionality of the user-movie interaction matrix
svd = TruncatedSVD(n_components=50, random_state=42)
user_movie_reduced = svd.fit_transform(user_movie_sparse)

# Output the shape of the reduced matrix
print("Reduced user-movie matrix shape:", user_movie_reduced.shape)

# Step 3: Normalize the reduced matrix (scaling each feature between 0 and 1)
normalizer = MinMaxScaler()
user_movie_reduced_normalized = normalizer.fit_transform(user_movie_reduced)

# Step 4: Fix the floating-point precision issue: Clip values between 0 and 1
user_movie_reduced_normalized = np.clip(user_movie_reduced_normalized, 0, 1)

# Validate that the normalized values are between 0 and 1
print(f"Min value after normalization: {user_movie_reduced_normalized.min()}")
print(f"Max value after normalization: {user_movie_reduced_normalized.max()}")

# Step 5: Check for missing values
if np.isnan(user_movie_reduced_normalized).any():
    print("Warning: Missing values detected. Handling missing values...")
    # Handling missing values (if any): fill NaNs with 0 (or use other imputation techniques)
    user_movie_reduced_normalized = np.nan_to_num(user_movie_reduced_normalized)

# Step 6: Split the normalized reduced matrix into training and testing sets (80% train, 20% test)
X_train, X_test = train_test_split(user_movie_reduced_normalized, test_size=0.2, random_state=42)

# Output the shape of the training and test sets to ensure they're as expected
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Save the preprocessed training and testing sets to NPZ
np.save('../data/processed/user_movie_reduced_train.npy', X_train)
np.save('../data/processed/user_movie_reduced_test.npy', X_test)

# Step 8: Cross-Validation Setup (Optional but recommended)
# Set up 5-fold cross-validation (adjust K as needed)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Example: Iterate through cross-validation splits (use during model evaluation)
for train_index, test_index in kf.split(X_train):
    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
    # You would train your model on X_train_cv and evaluate on X_test_cv here
    print(f"Training set for fold shape: {X_train_cv.shape}, Test set for fold shape: {X_test_cv.shape}")

# Step 9: Check matrix shape and consistency
print(f"Matrix reduced shape: {user_movie_reduced_normalized.shape}")
print(f"Matrix train-test split consistency: {X_train.shape[0] + X_test.shape[0] == user_movie_reduced_normalized.shape[0]}")

# %%
