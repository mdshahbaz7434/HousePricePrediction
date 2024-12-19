import pandas as pd
import numpy as np

# Load your original dataset
original_data = pd.read_csv("Housing.csv")  # Replace with the path to your dataset

# Set the number of rows for the synthetic dataset
num_rows = 25000

# Generate synthetic data while preserving distributions
synthetic_data = pd.DataFrame({
    "price": np.random.normal(original_data['price'].mean(), original_data['price'].std(), num_rows).astype(int),
    "area": np.random.normal(original_data['area'].mean(), original_data['area'].std(), num_rows).astype(int),
    "bedrooms": np.random.choice(
        original_data['bedrooms'].unique(),
        num_rows,
        p=original_data['bedrooms'].value_counts(normalize=True)
    ),
    "bathrooms": np.random.choice(
        original_data['bathrooms'].unique(),
        num_rows,
        p=original_data['bathrooms'].value_counts(normalize=True)
    ),
    "stories": np.random.choice(
        original_data['stories'].unique(),
        num_rows,
        p=original_data['stories'].value_counts(normalize=True)
    ),
    "mainroad": np.random.choice(
        original_data['mainroad'].unique(),
        num_rows,
        p=original_data['mainroad'].value_counts(normalize=True)
    ),
    "guestroom": np.random.choice(
        original_data['guestroom'].unique(),
        num_rows,
        p=original_data['guestroom'].value_counts(normalize=True)
    ),
    "basement": np.random.choice(
        original_data['basement'].unique(),
        num_rows,
        p=original_data['basement'].value_counts(normalize=True)
    ),
    "hotwaterheating": np.random.choice(
        original_data['hotwaterheating'].unique(),
        num_rows,
        p=original_data['hotwaterheating'].value_counts(normalize=True)
    ),
    "airconditioning": np.random.choice(
        original_data['airconditioning'].unique(),
        num_rows,
        p=original_data['airconditioning'].value_counts(normalize=True)
    ),
    "parking": np.random.choice(
        original_data['parking'].unique(),
        num_rows,
        p=original_data['parking'].value_counts(normalize=True)
    ),
    "prefarea": np.random.choice(
        original_data['prefarea'].unique(),
        num_rows,
        p=original_data['prefarea'].value_counts(normalize=True)
    ),
    "furnishingstatus": np.random.choice(
        original_data['furnishingstatus'].unique(),
        num_rows,
        p=original_data['furnishingstatus'].value_counts(normalize=True)
    )
})

# Clip numerical values to ensure realistic ranges
synthetic_data['price'] = synthetic_data['price'].clip(lower=original_data['price'].min(), upper=original_data['price'].max())
synthetic_data['area'] = synthetic_data['area'].clip(lower=original_data['area'].min(), upper=original_data['area'].max())

# Save the synthetic data to a CSV file
synthetic_data.to_csv("data.csv", index=False)

print("Synthetic dataset generated and saved as 'Synthetic_Housing_Data.csv'.")
