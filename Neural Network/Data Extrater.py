import pandas as pd

# Path to your combined CSV file
input_file = 'Merged Data.csv.'

# Read the combined dataset
df = pd.read_csv(input_file)

# Pick only the columns you need
# (Assuming these are the exact column names in the file)
columns_to_keep = [
    'BusinessTravel',
    'DistanceFromHome',
    'YearsAtCompany',
    'Age',
    'JobSatisfaction',
]
filtered_df = df[columns_to_keep]

# Save filtered results to a new CSV (or Excel)
filtered_df.to_csv('filtered_data.csv', index=False)
# Or if you prefer Excel:
# filtered_df.to_excel('filtered_data.xlsx', index=False)

print("Filtered data saved successfully.")
