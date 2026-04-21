import os
import numpy as np
import pandas as pd
# import your dependency:
# import esql 
# import ed

def get_weather_data(start_time, end_time, tn):
    weather_data = esql.select(
        ['win100_spd','d2','ssrd','tcc'], 
        start=start_time, end=end_time, NN=tn
    ).groupby('datetime').mean() 
    weather_data.index = pd.to_datetime(weather_data.index)
    weather_data = weather_data.resample('15T').interpolate(method='linear')
    weather_data['hour'] = weather_data.index.hour
    return weather_data

def get_orderbook_data(start_time, end_time):
    base_dir = "OrderBook_s1" # your data
    dfs = []
    for d in sorted(os.listdir(base_dir)):
        date_dir = os.path.join(base_dir, d)
        t2_dir = os.path.join(date_dir, "T2")
        if not os.path.isdir(t2_dir): continue

        for root, _, files in os.walk(t2_dir):
            for fn in files:
                if fn.endswith(".parquet"):
                    dfs.append(pd.read_parquet(os.path.join(root, fn)))

    if not dfs: return pd.DataFrame()
    trade_his = pd.concat(dfs, ignore_index=False, sort=False)
    
    try:
        trade_his.index = pd.to_datetime(trade_his.index)
    except:
        pass 
        
    def _last_level(x):
        if isinstance(x, str): x = eval(x)
        if isinstance(x, (list, tuple)) and len(x) > 0: return float(x[-1][0]), float(x[-1][1])
        return np.nan, np.nan
        
    def _first_level(x):
        if isinstance(x, str): x = eval(x)
        if isinstance(x, (list, tuple)) and len(x) > 0: return float(x[0][0]), float(x[0][1])
        return np.nan, np.nan

    trade_his[["bid_price", "bid_amount"]] = trade_his["bids"].apply(lambda x: pd.Series(_first_level(x)))
    trade_his[["ask_price", "ask_amount"]] = trade_his["asks"].apply(lambda x: pd.Series(_last_level(x)))
    trade_his["deal_price"] = (trade_his["bid_price"] + trade_his["ask_price"]) / 2
    
    return trade_his[(trade_his.index >= start_time) & (trade_his.index <= end_time)]

def get_realprice_data(start_time, end_time):
    price_data = ed.pull(['da', 'rt'], start=start_time, end=end_time)
    price_data['time'] = pd.to_datetime(price_data.index)
    price_data.set_index('time', inplace=True)
    return price_data

def _load_prediction_parquet(start_time, end_time, path_template):
    dfs = []
    for d in pd.date_range(start_time, end_time, freq="D"):
        p = path_template.format(date_str=d.strftime('%Y%m%d'))
        if os.path.exists(p): 
            dfs.append(pd.read_parquet(p))
    
    if not dfs: return pd.DataFrame()
    res = pd.concat(dfs)
    res.index = pd.to_datetime(res.index)
    return res[(res.index >= start_time) & (res.index <= end_time)]

def get_rt_d2_prediction(start_time, end_time):
    return _load_prediction_parquet(start_time, end_time, "N2rt.parquet") # your data

def get_da_prediction(start_time, end_time, is_d2=False):
    folder = "N2" if is_d2 else "N1"
    return _load_prediction_parquet(start_time, end_time, f"da.parquet") # your data

def get_rt_da_prediction(start_time, end_time):
    return _load_prediction_parquet(start_time, end_time, "rt.parquet") # your data

def prepare_base_dataframe(start_time, end_time):
    print(f"--- Fetching & Aligning Data ({start_time} to {end_time}) ---")
    
    df_weather_d2 = get_weather_data(start_time, end_time, tn=2)
    df_orderbook = get_orderbook_data(start_time, end_time)
    df_real = get_realprice_data(start_time, end_time)
    
    df_pred_rt_d2 = get_rt_d2_prediction(start_time, end_time)
    df_pred_rt_da = get_rt_da_prediction(start_time, end_time) 
    df_pred_da_da = get_da_prediction(start_time, end_time, is_d2=False) 
    
    df_pred_rt_d2 = df_pred_rt_d2.rename(columns={'rt': 'pred_rt_d2'})
    df_pred_rt_da = df_pred_rt_da.rename(columns={'rt': 'pred_rt_da'})
    df_pred_da_da = df_pred_da_da.rename(columns={'da': 'pred_da_da'})
    
    df = df_real.join(df_orderbook['deal_price'], how='left') \
                .join(df_pred_rt_d2['pred_rt_d2'], how='left') \
                .join(df_pred_rt_da['pred_rt_da'], how='left') \
                .join(df_pred_da_da['pred_da_da'], how='left')
                
    df = df.ffill().bfill()
    
    df['true_rt_d2_spread'] = df['rt'] - df['deal_price']
    df['true_rt_da_spread'] = df['rt'] - df['da']
    
    df['d2_pred_spread'] = df['pred_rt_d2'] - df['deal_price']
    df['da_pred_spread'] = df['pred_rt_da'] - df['pred_da_da']
    
    rolling_window = 96 * 7 
    d2_error = df['pred_rt_d2'] - df['rt']
    da_error = df['pred_rt_da'] - df['rt']
    
    df['d2_confidence'] = 1.0 / (d2_error.rolling(rolling_window).std() + 1e-5)
    df['da_confidence'] = 1.0 / (da_error.rolling(rolling_window).std() + 1e-5)
    
    df['d2_confidence'] = (df['d2_confidence'] - df['d2_confidence'].min()) / (df['d2_confidence'].max() - df['d2_confidence'].min() + 1e-5)
    df['da_confidence'] = (df['da_confidence'] - df['da_confidence'].min()) / (df['da_confidence'].max() - df['da_confidence'].min() + 1e-5)

    weather_cols = ['win100_spd','d2','ssrd','tcc','hour']
    df = df.join(df_weather_d2[weather_cols], how='left')
    
    df = df.ffill().bfill().dropna()
    print(f"Data alignment complete. Valid periods: {len(df)}")
    return df

def augment_training_data(df, **kwargs):
    """
    i.e. : Add your custom data augmentation logic here. note some features, like "hourofday ","dayofweek"
    - Weather feature noise injection (Gaussian).
    - Prediction spread perturbation.
    - Bootstrapping or time-shifting.
    """
    print("[Info] Using custom data augmentation module...")
    # Add your logic
    return df
