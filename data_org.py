import pandas as pd
import numpy as np

# Set the number of rows for the synthetic dataset
num_rows = 25000

# Base price and area range
base_price = 5000000
area_mean = 5000
area_std = 1500

# Generate synthetic data
synthetic_data = pd.DataFrame({
    "area": np.random.normal(area_mean, area_std, num_rows).astype(int),
    "bedrooms": np.random.choice([2, 3, 4, 5], num_rows, p=[0.2, 0.5, 0.2, 0.1]),
    "bathrooms": np.random.choice([1, 2, 3, 4], num_rows, p=[0.4, 0.4, 0.15, 0.05]),
    "stories": np.random.choice([1, 2, 3], num_rows, p=[0.5, 0.4, 0.1]),
    "mainroad": np.random.choice(["yes", "no"], num_rows, p=[0.7, 0.3]),
    "guestroom": np.random.choice(["yes", "no"], num_rows, p=[0.3, 0.7]),
    "basement": np.random.choice(["yes", "no"], num_rows, p=[0.4, 0.6]),
    "hotwaterheating": np.random.choice(["yes", "no"], num_rows, p=[0.2, 0.8]),
    "airconditioning": np.random.choice(["yes", "no"], num_rows, p=[0.6, 0.4]),
    "parking": np.random.choice([0, 1, 2, 3], num_rows, p=[0.3, 0.4, 0.2, 0.1]),
    "prefarea": np.random.choice(["yes", "no"], num_rows, p=[0.5, 0.5]),
    "furnishingstatus": np.random.choice(["furnished", "semi-furnished", "unfurnished"], num_rows, p=[0.4, 0.4, 0.2])
})

# Calculate price based on rules
def calculate_price(row):
    price = base_price
    
    # Adjust for area
    price += (row["area"] - area_mean) * 100  # Larger area increases price

    # Bedrooms and Bathrooms
    if row["bedrooms"] == 3:
        price += 500000  # Higher price for 3 bedrooms
    price += row["bathrooms"] * 200000  # Each bathroom adds value

    # Stories
    if row["stories"] > row["bathrooms"]:  # Double story for each bathroom
        price += 300000

    # Main Road
    if row["mainroad"] == "yes":
        price += 300000

    # Guestroom
    if row["guestroom"] == "yes":
        price += 200000

    # Basement
    if row["basement"] == "yes":
        price += 150000

    # Hot Water Heating
    if row["hotwaterheating"] == "yes":
        price += 100000

    # Air Conditioning
    if row["airconditioning"] == "yes":
        price += 250000

    # Parking
    price += row["parking"] * 100000

    # Preferred Area
    if row["prefarea"] == "yes":
        price += 400000

    # Furnishing Status
    if row["furnishingstatus"] == "furnished":
        price += 300000
    elif row["furnishingstatus"] == "unfurnished":
        price -= 200000

    return price

# Apply price calculation
synthetic_data["price"] = synthetic_data.apply(calculate_price, axis=1)

# Clip unrealistic values
synthetic_data["price"] = synthetic_data["price"].clip(lower=2000000, upper=15000000)
synthetic_data["area"] = synthetic_data["area"].clip(lower=1500, upper=20000)

# Save the dataset
synthetic_data.to_csv("Realistic_Housing_Data.csv", index=False)

print("Realistic synthetic dataset generated and saved as 'Realistic_Housing_Data.csv'.")
