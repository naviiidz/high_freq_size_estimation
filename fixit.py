import pandas as pd

# Load the CSV file
df = pd.read_csv('pe_signals.csv')

# Add brackets only if not present
df['Radius'] = df['Radius'].apply(lambda x: f'[{x}]' if not str(x).startswith('[') else x)

# Save the modified DataFrame
df.to_csv('updated_pe_signals.csv', index=False)

