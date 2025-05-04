import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
input_csv_path = r"C:\Users\mchak\Desktop\UNSW_NB15_training-set.csv"  # Replace with the path to your dataset
output_csv_path = r"C:\Users\mchak\Desktop\output.csv"  # Replace with the path to save the preprocessed dataset
df = pd.read_csv(input_csv_path)
# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (you can modify this based on your requirements)
df.fillna(0, inplace=True)

# Categorical columns to one-hot encode
categorical_cols = ['proto', 'service', 'state']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode the target column 'label' if necessary
if 'label' in df.columns:
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

# Exclude specific columns like 'id' and 'label' from scaling
columns_to_exclude = ['id', 'label']  # Add 'id' here if it's in your dataset
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_exclude)

# Scale numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the preprocessed data to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Preprocessed data saved to: {output_csv_path}")
