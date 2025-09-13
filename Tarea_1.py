import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import anderson
import seaborn as sns

pd.set_option('display.max_columns', None)


### Data manipulation ###

# Read data
df = pd.read_csv('./data/data.csv')

# Separate location and data information
df_data = df.iloc[9:].copy()
df_loc = df.iloc[:8].copy()

# Clean location information for display purposes
df_loc = df_loc.transpose()
df_loc.columns = df_loc.iloc[0]
df_loc = df_loc.drop(df_loc.index[0])
df_loc.rename(columns={'Site name': 'Index_'}, inplace=True)
df_loc['years_with_data'] = (df_loc['Last year CE'].astype(int)
                             - df_loc['First year CE'].astype(int)
                             + 1
                             )
df_loc['Site  name'] = df_loc['Site  name'].str[:23]
df_loc.reset_index(inplace=True)
df_loc.rename(columns={'index': 'site_code',
                       'Site  name': 'site_name'},
              inplace=True
              )

# Plot years with data according to location
fig = px.scatter_geo(df_loc,
                    lat=df_loc.Latitude,
                    lon=df_loc.Longitude,
                    color='site_name',
                    size="years_with_data",
                     )
fig.update_geos(
    center={"lat": df_loc["Latitude"].astype(float).mean(),
            "lon": df_loc["Longitude"].astype(float).mean()},
    projection_scale=4
)

fig.update_layout(title_text='Number of years with data according to location')
fig.show()

print(df_loc)

# Clean data information
df_data.rename(columns={'Site Code': 'year'}, inplace=True)
df_data.set_index('year', inplace=True)
df_data = df_data.transpose().reset_index()
df_data.rename(columns={'index': 'site_code'}, inplace=True)
df_data = df_data.melt(id_vars=['site_code'], var_name='Year', value_name='Value')
df_data.sort_values(by=['site_code', 'Year'], inplace=True, ignore_index=True)

# Identify problematic entries in 'Value' column
mask_invalid = pd.to_numeric(df_data['Value'].str.strip(), errors='coerce').isna()
errores = df_data.loc[mask_invalid, 'Value']
print("Problematic entries in registered 'Value' column:")
l_problematic_entries = errores.unique().tolist()
print(l_problematic_entries)

# This will be done since all problematic entries are some form of NaN
df_data['Value'] = df_data['Value'].replace(
    ['nan', 'NaN', 'NAN'], np.nan
)


### Number of missing values per site ###

# After EDA, we found that 'NA' strings are used to denote missing data
df_data['Value'] = pd.to_numeric(df_data['Value'].str.strip(), errors='coerce')

#Fix errors with site strings
df_data['site_code']= df_data.site_code.str.strip()

# Compare number of data points per site with number of years with data
df_data_value_counts = df_data.groupby('site_code', as_index=False).count()
df_data_value_counts.columns = [col if col == 'site_code'
                                else col + '_count'
                                for col in df_data_value_counts.columns
                                ]

df_loc['site_code'] = df_loc.site_code.str.strip()

df_data_value_counts = df_data_value_counts.merge(
    					df_loc[['site_code', 'years_with_data']],
                        on='site_code', how='left'
                        )
df_data_value_counts['missing_data'] = (df_data_value_counts['Value_count'] 
                                        < df_data_value_counts['years_with_data'])*1

print("Number of sites with missing data: "
      f"{df_data_value_counts['missing_data'].sum()} out of "
      f"{df_data_value_counts.shape[0]} total sites")

df_data_value_counts['percentage_missing'] = (1
                                              - (df_data_value_counts['Value_count']
                                                 /df_data_value_counts['years_with_data'])
                                                 )*100


### Visual review of sites with missing data ###

# Segregate sites according to missing data
df_significant_missing_data = df_data_value_counts[
                                    df_data_value_counts.percentage_missing > 6
                                    ].reset_index(drop=True)
df_low_missing_data = df_data_value_counts[
                                    (df_data_value_counts.percentage_missing <= 6)
                                    & (df_data_value_counts.percentage_missing > 0)
                                    ].reset_index(drop=True)
df_more_data = df_data_value_counts[
                    df_data_value_counts.percentage_missing < 0].reset_index(drop=True)
print('Places with significant missing data')
print(df_significant_missing_data)
print('Places with more missing data')
print(df_more_data)

# Define list with respective cases
l_hard_review = df_significant_missing_data['site_code'].tolist()
l_more_data = df_more_data['site_code'].tolist()
l_low_review = df_low_missing_data['site_code'].tolist()
l_review = l_hard_review + l_low_review + l_more_data
l_no_problem = list(df_data.site_code.unique())
l_no_problem = [i for i in l_no_problem if i not in l_review]
# Figure for 1 row and 2 columns for hard review
fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Site code: {i}" for i in l_hard_review])

# Add figures
for idx, i in enumerate(l_hard_review):
    aux = df_data[df_data['site_code'] == i].reset_index(drop=True)
    fig.add_trace(
        go.Scatter(x=aux['Year'], y=aux['Value'], mode='lines', name=f"Site {i}"),
        row=1, col=idx+1
    )

# Fix layout
fig.update_layout(title_text="Sitios con gran cantidad de faltantes.", showlegend=False)
fig.show()

# Figure for 4 row and 2 columns for low review
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[f"Site code: {i}" for i in l_low_review]
)

# Add figures
for idx, i in enumerate(l_low_review):
    aux = df_data[df_data['site_code'] == i].reset_index(drop=True)
    
    # Calcular fila y columna
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    fig.add_trace(
        go.Scatter(x=aux['Year'], y=aux['Value'], mode='lines', name=f"Site {i}"),
        row=row, col=col
    )

# Ajustar layout
fig.update_layout(
    title_text="Comparación de sitios con pocos datos faltantes.",
    showlegend=False,
    height=1600, width=1600
)
fig.show()


### Fix missing data accordingly ###

# For REN data will be kept after 1751
# For FON data will be kept after 1829

df_data['Year'] = df_data.Year.astype(int)

df_data_ren = df_data.copy()
df_data_ren = df_data_ren[(df_data_ren.site_code == 'REN')].reset_index(drop=True)
df_data_fon = df_data.copy()
df_data_fon = df_data_fon[df_data_fon.site_code == 'FON'].copy().reset_index(drop=True)
row_indexer_ren = df_data_ren.Year < 1751
row_indexer_fon = df_data_fon.Year < 1829

df_data_ren.loc[row_indexer_ren, 'Value'] = np.nan
df_data_fon.loc[row_indexer_fon, 'Value'] = np.nan

df_data_mod = df_data.copy()
df_data_mod = df_data_mod[~df_data_mod.site_code.isin(['REN', 'FON', 'COL'])].reset_index(drop=True)
df_data_mod = pd.concat([df_data_mod, df_data_ren, df_data_fon], ignore_index=True)
df_data_interpol = df_data_mod.copy()
# Sort valued for agrupation and data interpolation
df_data_interpol = df_data_interpol.sort_values(['site_code', 'Year'])
# Interpolate within each site
df_data_interpol['Value'] = (
    df_data_interpol
    .groupby('site_code', group_keys=False)['Value']
    .apply(lambda s: s.interpolate(method='linear', limit_area='inside'))
)

# Alternative method to fill missing data with mean of each site
df_data_mean_imputed = df_data_mod.copy()
df_data_mean_imputed['Value'] = (
    df_data_mean_imputed
    .groupby('site_code')['Value']
    .transform(lambda s: s.fillna(s.mean()))
)


### Show manipulated data by interpolations ###

l_fixed_cases = l_low_review + l_hard_review


# Figure with 5 rows y 2 columns
fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=[f"Site code: {i}" for i in l_fixed_cases]
)

# Add figures to subplot
for idx, i in enumerate(l_fixed_cases):
    aux = df_data_interpol[df_data_interpol['site_code'] == i].reset_index(drop=True)
    
    # Calcular fila y columna (enumerate empieza en 0)
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    fig.add_trace(
        go.Scatter(x=aux['Year'], y=aux['Value'], mode='lines', name=f"Site {i}"),
        row=row, col=col
    )

# AAdjust layout
fig.update_layout(
    title_text="Comparación de sitios (10 con imputación)",
    showlegend=False,
    height=2000, width=1600
)

fig.show()


### Check for outliers globally ###

## Using normal assumption ##

# Visualize values in all sites
px.histogram(df_data, x='Value', title='Distribution of Values', nbins=60).show()

# Do Anderson darling test to see if the data matches normal distribution
result = anderson(df_data['Value'].dropna(), dist='norm' )
print('Result :', result)

# Once normal distribution can be associated standarize value to check for possible outliers
df_data['standarized_value'] = (df_data.Value - df_data.Value.mean()) / df_data.Value.std(ddof=0)

# Show outliers
print(df_data[df_data.standarized_value.abs() > 3])

# plot standarized values
px.histogram(df_data, x='standarized_value', title='Distribution of Values', nbins=60).show()


## Now using hat matrix ##
X = df_data_interpol['Year']
y = df_data_interpol['Value']

# Añadir intercepto
X1 = np.column_stack([np.ones(X.shape[0]), X])

# Calcular beta con mínimos cuadrados
beta_hat, *_ = np.linalg.lstsq(X1, y, rcond=None)

# Calcular matriz sombrero H
H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
leverages = np.diag(H)

# Regla práctica de corte
n, p = X1.shape
threshold = 2*p/n

# --- Visualización ---
plt.figure(figsize=(8,5))
plt.scatter(X, y, c="blue", alpha=0.6, label="Datos")
plt.plot(X, X1 @ beta_hat, c="red", label="Recta ajustada")

# Resaltar puntos con leverage alto
outliers = leverages > threshold


## Using normal assumption on interpolated data to check for changes ##

# Visualize values in all sites
px.histogram(df_data_interpol, x='Value', title='Distribution of Values', nbins=60).show()

# Do Anderson darling test to see if the data matches normal distribution
result = anderson(df_data_interpol['Value'].dropna(), dist='norm' )
print('Result :', result)

# Once normal distribution can be associated standarize value to check for possible outliers
df_data_interpol['standarized_value'] = (df_data_interpol.Value - df_data_interpol.Value.mean()) / df_data_interpol.Value.std(ddof=0)

# Show outliers
print(df_data_interpol[df_data_interpol.standarized_value.abs() > 3])

# plot standarized values
px.histogram(df_data_interpol, x='standarized_value', title='Distribution of Values', nbins=60).show()

## By tree species and location ##

df_data_interpol_species = df_data_interpol.merge(df_loc[['site_code', 'Species']],
                                                  on='site_code',
                                                  how='left')

# Lista de especies únicas
species_list = list(df_data_interpol_species['Species'].unique())

# Crear subplots con 3 filas y 2 columnas
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[f"Species: {sp}" for sp in species_list]
)

# Iterar sobre especies
for idx, sp in enumerate(species_list):
    aux = df_data_interpol_species[df_data_interpol_species['Species'] == sp].reset_index(drop=True)
    
    # Calcular fila y columna
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    # Agregar scatter por cada sitio dentro de la especie
    for site in aux['site_code'].unique():
        site_aux = aux[aux['site_code'] == site]
        fig.add_trace(
            go.Scatter(
                x=site_aux['Year'], y=site_aux['Value'],
                mode='markers',
                name=f"{sp} - {site}",
                
            ),
            row=row, col=col
        )

# Ajustar layout
fig.update_layout(
    title_text="Homogeneidad entre especies",
    height=1600, width=1600
)

fig.show()


### Standarization ###

# Standarization of SER and POE Site codes
df_caz = df_data[(df_data['site_code'] == 'CAZ') & (df_data['Value'].notna())].reset_index(drop=True)
df_caz['min_max'] = (df_caz['Value'] - df_caz['Value'].min()) / (df_caz['Value'].max() - df_caz['Value'].min())
df_caz['z_score'] = (df_caz['Value'] - df_caz['Value'].mean()) / df_caz['Value'].std(ddof=0)

df_poe = df_data[(df_data['site_code'] == 'POE') & (df_data['Value'].notna())].reset_index(drop=True)
df_poe['min_max'] = (df_poe['Value'] - df_poe['Value'].min()) / (df_poe['Value'].max() - df_poe['Value'].min())
df_poe['z_score'] = (df_poe['Value'] - df_poe['Value'].mean()) / df_poe['Value'].std(ddof=0)

print(df_caz)

## Boxplot ##

# Combine data for plotting
df_combined = pd.concat([df_caz, df_poe], ignore_index=True)

# Melt the min-max DataFrame to long format for plotly
df_long_mm = df_combined.melt(id_vars=['site_code'], value_vars=['min_max'],
                           var_name='Standardization', value_name='Standardized_Value')

# Create boxplot of min-max standardization
fig = px.box(df_long_mm, x='site_code', y='Standardized_Value',
             title='Min-Max standardization Boxplot for CAZ and POE',)
fig.show()

# Melt the z-score DataFrame to long format for plotly
df_long_z = df_combined.melt(id_vars=['site_code'], value_vars=['z_score'],
                           var_name='Standardization', value_name='Standardized_Value')

# Create boxplot of z-score standardization
fig = px.box(df_long_z, x='site_code', y='Standardized_Value',
             title='Z-score standardization Boxplot for CAZ and POE',)
fig.show()

# Histogram of standardized values
# --- Histograms ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# z-score CAZ histogram
sns.histplot(df_caz['z_score'].dropna(), kde=False, ax=axes[0], color="skyblue")
axes[0].set_title(f"CAZ z-score histogram")

# z-score POE histogram
sns.histplot(df_poe['z_score'].dropna(), kde=False, ax=axes[1], color="orange")
axes[1].set_title(f"POE z-score histogram")

plt.tight_layout()
plt.show()


## Scatter plot and histograms for original data ##

fig = px.scatter(
    df_combined,
    x='Year',
    y='Value',
    color='site_code',
    title='Scatter plot for CAZ and POE'
)
fig.show()

# Merge by year
df_merge = pd.merge(
    df_caz[['Year', 'Value']],
    df_poe[['Year', 'Value']],
    on='Year',
    how='outer',
    suffixes=('_CAZ', '_POE')
)

# Scatter plot
fig = px.scatter(
    df_merge,
    x='Value_CAZ',
    y='Value_POE',
    title='POE vs CAZ by year',
    labels={'Value_CAZ': 'CAZ Value', 'Value_POE': 'POE Value'}
)
fig.show()

# --- Histograms and kernel density plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# CAZ Histogram
sns.histplot(df_caz['Value'].dropna(), kde=False, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title(f"CAZ histogram")
# CAZ kernel density
sns.kdeplot(df_caz['Value'].dropna(), ax=axes[0, 1], color="green", fill=True)
axes[0, 1].set_title(f"CAZ kernel density")  

# POE Histogram
sns.histplot(df_poe['Value'].dropna(), kde=False, ax=axes[1, 0], color="orange")
axes[1, 0].set_title(f"POE histogram")
# POE kernel density
sns.kdeplot(df_poe['Value'].dropna(), ax=axes[1, 1], color="red", fill=True)
axes[1, 1].set_title(f"POE kernel density")
    
plt.tight_layout()
plt.show()