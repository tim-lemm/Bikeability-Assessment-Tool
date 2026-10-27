import pandas as pd

df_donnees_2016 = pd.read_csv("data/BTT_TD_NAV2A_2016.csv", sep=";", encoding="latin1")
df_donnees_2016['CODGEO'] = pd.to_numeric(df_donnees_2016['CODGEO'], errors='coerce')
df_donnees_2016 = df_donnees_2016[df_donnees_2016["CODGEO"].between(69000, 70000)]

print(df_donnees_2016.head())

df_modes_lyon = df_donnees_2016.groupby('TRANS')['NB'].sum().reset_index()
df_modes_lyon['PART'] = (df_modes_lyon['NB']/df_modes_lyon['NB'].sum())*100
print(df_modes_lyon)
