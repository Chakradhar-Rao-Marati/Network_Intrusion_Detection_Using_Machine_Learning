import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Load the dataset
input_csv_path = r"C:\Users\mchak\Desktop\New folder\output.csv"  # Replace with the path to your dataset
output_csv_path = r"C:\Users\mchak\Desktop\New folder\ro_pca_new.csv"  # Replace with the path to save the processed dataset
df = pd.read_csv(input_csv_path)

# Separate features and target
X = df.drop(columns=['label'])  # Features (excluding 'label')
y = df['label']  # Target variable (label)

# Identify non-numeric columns (categorical data) and apply encoding
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes  # Label Encoding

# Standardize numerical features (important for PCA)
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Random Oversampling (to balance the dataset)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Apply PCA (retain 95% of the variance)
pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
X_pca = pca.fit_transform(X_resampled)

# Determine the original feature with the highest contribution to each principal component
components = pca.components_  # Eigenvectors (principal components)
feature_names = X.columns  # Original feature names

# Get the top contributing feature for each principal component
top_features = [feature_names[np.argmax(abs(components[i]))] for i in range(components.shape[0])]

# Create DataFrame with meaningful column names
processed_df = pd.DataFrame(X_pca, columns=top_features)
processed_df['label'] = y_resampled.reset_index(drop=True)

# Save the processed dataset
processed_df.to_csv(output_csv_path, index=False)
print(f"Processed data saved to: {output_csv_path}")

# Output explained variance ratio of PCA components
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each principal component:", explained_variance)
print(f"Number of components retained: {X_pca.shape[1]}")
