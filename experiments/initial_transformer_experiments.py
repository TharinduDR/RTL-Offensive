import pandas as pd

data_df = pd.read_json('data/rtl_comments_clean_2022-10.json')
data_df_clean = data_df[data_df.status.isin(["Published", "Archived"])]
data_df_clean['date_created'] = pd.to_datetime(data_df_clean['date_created'])

# Extract year, month, day of the week, and hour
data_df_clean['year'] = data_df_clean['date_created'].dt.year
data_df_clean['month'] = data_df_clean['date_created'].dt.month
data_df_clean['day_of_week'] = data_df_clean['date_created'].dt.strftime('%A') # Extract day of the week as full name
data_df_clean['hour'] = data_df_clean['date_created'].dt.hour



