import pandas as pd

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
csv_files = [str(elt) + '.csv' for elt in years]
total = []

for file in csv_files: # Combine csv
    df = pd.read_csv(file, index_col = None, header = 0)
    total.append(df)
    print(f"{file} - appended")

print()

appended_df = pd.concat(total, ignore_index=False)
appended_df.rename(columns={'Unnamed: 0':'Race ID'}, inplace=True)

print(appended_df.head())


id_counter = 0
races = {}

unique_races = appended_df['Country'].unique()
races = {race: idx for idx, race in enumerate(unique_races, start=id_counter)}

appended_df['Race ID'] = appended_df['Country'].map(races)

appended_df.to_csv('prenorm.csv', index=False)


