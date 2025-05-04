## A couple of functions of generating new features
import numpy as np
import pandas as pd

class Location_Transformer():
    def fit(self, X):
        file_path = 'CP.csv'  
        cp = pd.read_csv(file_path)
        cp['name'] = cp['name'].str.lower()

        ##Create two dictionaries for looking up
        self.dict_la = {key: value for key, value in zip(cp['name'], cp['latitude'])}
        self.dict_lg = {key: value for key, value in zip(cp['name'], cp['longitude'])}

    def Bene_transform(self, X_Bene):
        list_a=[]
        list_b=[]
        for x in X_Bene:
            if x.lower() in self.dict_la:
                list_a.append(self.dict_la[x.lower()])
                list_b.append(self.dict_lg[x.lower()])
            else:
                list_a.append(0)
                list_b.append(0)
        return pd.DataFrame({'Bene_lat': list_a, 'Bene_lg': list_b})
    
    def Sender_transform(self, X_Sender):
        list_a=[]
        list_b=[]
        for x in X_Sender:
            if x.lower() in self.dict_la:
                list_a.append(self.dict_la[x.lower()])
                list_b.append(self.dict_lg[x.lower()])
            else:
                list_a.append(0)
                list_b.append(0)
        return pd.DataFrame({'Sender_lat': list_a, 'Sender_lg': list_b})
    
    def Loc_Cor_transform(self, X):
        self.fit(X)
        X['Bene_Country'] = X.Bene_Country.str.replace('-', ' ')
        X['Sender_Country'] = X.Sender_Country.str.replace('-', ' ')
        df_sc=self.Sender_transform(X.Sender_Country)
        df_bc=self.Bene_transform(X.Bene_Country)
        X = pd.concat([X, df_sc, df_bc], axis=1)
        X = self.haversine(X)
        return X
    
    def haversine(self, X):
        # Convert latitude and longitude from degrees to radians
        # Convert decimal degrees to radians 
        X['lat1'], X['lon1'], X['lat2'], X['lon2'] = map(np.radians, [X.Sender_lat, X.Sender_lg,         X.Bene_lat, X.Bene_lg])

        # Haversine formula 
        X['dlon'] = X['lon2'] - X['lon1'] 
        X['dlat'] = X['lat2'] - X['lat1'] 

        X['a'] = np.sin(X['dlat'] / 2)**2 + np.cos(X['lat1']) * np.cos(X['lat2'])*np.sin(X['dlon'] / 2)**2
        X['c'] = 2 * np.arctan2(np.sqrt(X['a']), np.sqrt(1 - X['a'])) 

        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371 
        X['dist'] = r * X['c']
        cols_to_drop = ['lat1','lon1','lat2','lon2','dlon','dlat','a','c']
        X.drop(columns=cols_to_drop,inplace = True)
        return X
        
    

class group_transformer():
    def __init__(self, binwidth, fn):
        self.bin = binwidth
        self.fn = fn
        
    def fit(self,X):
        self.X=X
        
    def transform(self,X):
        ##Generate interval features for numerical values
        bin_edges = list(range(0, int(X[self.fn].max()) + self.bin, self.bin))
        ng = self.fn+'_group'
        X[ng] = pd.cut(X[self.fn], bins=bin_edges, right=False, include_lowest=True)
        return X
    

def individual_behavior(df, name):
    # name can be "Sender_Id", "Bene_Id"
    # Count transactions per day for each sender_id
    count_name = name+'_dc'
    daily_transactions = df.groupby([name, 'date']).size().reset_index(name=count_name)

    # Merge back to the original dataframe
    df = df.merge(daily_transactions[[name, 'date', count_name]], on=[name, 'date'], how='left')

    # Sort by sender_id and date
    daily_transactions.sort_values(by=[name, 'date'], inplace=True)

    # Calculate the difference in daily transaction counts
    count_diff_name = name+'_cdf'
    increase_rate_name = name+'_incr'
    daily_transactions[count_diff_name] = daily_transactions.groupby(name)[count_name].diff().fillna(0)
    daily_transactions[increase_rate_name] = daily_transactions.groupby(name)[count_name].pct_change().fillna(0)
    # Merge back with original DataFrame
    df = df.merge(daily_transactions[[name, 'date', count_diff_name, increase_rate_name]], on=[name, 'date'], how='left')

    # Calculate the cumulative average up to the day before each transaction
    daily_transactions['cumulative_sum'] = daily_transactions.groupby(name)[count_name].cumsum().shift(1)
    daily_transactions['transaction_days'] = daily_transactions.groupby(name).cumcount()
    
    cum_avg_name=name+'_cumavg'
    daily_transactions[cum_avg_name] = daily_transactions['cumulative_sum'] / daily_transactions['transaction_days']
    daily_transactions[cum_avg_name] = daily_transactions[cum_avg_name].fillna(0)
    daily_transactions[cum_avg_name] = daily_transactions[cum_avg_name].replace([-np.inf, np.inf], 0)
    # Calculate the difference
    avg_diff_name=name+'_avgdif'
    daily_transactions[avg_diff_name] = daily_transactions[count_name]-daily_transactions[cum_avg_name].fillna(0)
    daily_transactions[avg_diff_name] = daily_transactions[avg_diff_name].replace([-np.inf, np.inf], 0)

    # Calculate the difference
    avg_incr_name=name+'_avgincr'
    daily_transactions[avg_incr_name] = ((daily_transactions[count_name] -   daily_transactions[cum_avg_name])/daily_transactions[cum_avg_name]).fillna(0)
    daily_transactions[avg_incr_name] = daily_transactions[avg_incr_name].replace([-np.inf, np.inf], 0)

    # Calculate days since initial
    initial_dates = daily_transactions.groupby(name)['date'].min().rename('initial_date')

    # Merge the initial transaction date back into the DataFrame
    daily_transactions = daily_transactions.merge(initial_dates, on=name, how='left')

    # Calculate the number of days since the initial transaction
    day_sin_name=name+'_daysin'
    daily_transactions[day_sin_name] = (daily_transactions['date'] - daily_transactions['initial_date']).dt.days

    # Merge back with original DataFrame
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' format is consistent
    daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])

    df = df.merge(daily_transactions[[name, 'date', avg_diff_name]], on=[name, 'date'], how='left')

    df = df.merge(daily_transactions[[name, 'date', avg_incr_name]], on=[name, 'date'], how='left')

    df = df.merge(daily_transactions[[name, 'date', day_sin_name]], on=[name, 'date'], how='left')

    df = df.merge(daily_transactions[[name, 'date', cum_avg_name]], on=[name, 'date'], how='left')
    return df

def time_difference(df,name):
    df.sort_values(by=[name,'Time_step'], inplace=True)
    time_diff_name = name+'_timediff'
    df['shifted_time'] = df.groupby(name)['Time_step'].shift()
    df[time_diff_name] = (df['Time_step']-df['shifted_time']).fillna(pd.Timedelta(seconds=0))
  #  df[time_diff_name] = df.groupby(name)['Time_step'].diff().fillna(pd.Timedelta(seconds=0)) 
    df[time_diff_name] = df[time_diff_name].dt.total_seconds()/60.0
    df.drop(['shifted_time'], axis=1, inplace=True)
    return df
    
def haversine_vectorized(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
    return c * r

def sender_geo_velocity(df,name):
    # Sort the DataFrame by 'sender_id' and 'time_stamp'
    df.sort_values(by=[name, 'Time_step'], inplace=True)

    # Shift the latitude and longitude columns
    df['shifted_lon_sac'] = df.groupby(name)['Sender_lg'].shift()
    df['shifted_lat_sac'] = df.groupby(name)['Sender_lat'].shift()

    # Apply the vectorized haversine function
    geo_dist_name=name+'_geodist'
    df[geo_dist_name] = haversine_vectorized(df['shifted_lon_sac'], df['shifted_lat_sac'], df['Sender_lg'], df['Sender_lat'])
    
    # Drop the shifted columns if they are no longer needed
    df.drop(['shifted_lon_sac', 'shifted_lat_sac'], axis=1, inplace=True)
    return df

def bene_geo_velocity(df,name):
    # Sort the DataFrame by 'sender_id' and 'time_stamp'
    df.sort_values(by=[name, 'Time_step'], inplace=True)

    # Shift the latitude and longitude columns
    df['shifted_lon_sac'] = df.groupby(name)['Bene_lg'].shift()
    df['shifted_lat_sac'] = df.groupby(name)['Bene_lat'].shift()

    # Apply the vectorized haversine function
    geo_dist_name=name+'_geodist'
    df[geo_dist_name] = haversine_vectorized(df['shifted_lon_sac'], df['shifted_lat_sac'], df['Bene_lg'], df['Bene_lat'])
    
    # Drop the shifted columns if they are no longer needed
    df.drop(['shifted_lon_sac', 'shifted_lat_sac'], axis=1, inplace=True)
    return df