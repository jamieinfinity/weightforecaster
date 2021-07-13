import shutil
import datetime
import numpy as np
import pandas as pd
import configparser
import fitbit
import myfitnesspal
from nokia import NokiaApi, NokiaCredentials  # Withings

REFRESH_FIRST_DATE_LOOKBACK_DAYS = 7


def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def get_latest_date(column, df):
    df_filtered = pd.DataFrame(df.loc[df[column] > 0], copy=True)
    df_filtered.sort_index(ascending=False, inplace=True)
    return df_filtered.iloc[0].name


def get_target_date_endpoints(column, df):
    today = datetime.date.today()
    today = datetime.datetime.combine(today, datetime.datetime.min.time())
    latest_date = get_latest_date(column, df)
    first_date = latest_date - datetime.timedelta(days=REFRESH_FIRST_DATE_LOOKBACK_DAYS)
    first_date = datetime.datetime.combine(first_date.date(), datetime.datetime.min.time())
    last_date = today
    return [first_date, last_date]


def get_target_date_range(column, df):
    [first_date, last_date] = get_target_date_endpoints(column, df)
    target_dates = pd.date_range(first_date, last_date).values
    return [pd.to_datetime(d) for d in target_dates]


def insert_values(date_values, column, df):
    df_updated = pd.DataFrame(df, copy=True)
    for dv in date_values:
        df_updated.at[dv[0], column]=dv[1]
    df_updated.sort_index(inplace=True)
    return df_updated


def persist_fitbit_refresh_token(token_dict, cfg_file):
    parser = configparser.ConfigParser()
    parser.read(cfg_file)
    parser.set('fitbit', 'access_token', token_dict['access_token'])
    parser.set('fitbit', 'refresh_token', token_dict['refresh_token'])
    parser.set('fitbit', 'expires_at', "{:.6f}".format(token_dict['expires_at']))
    with open(cfg_file, 'w') as configfile:
        parser.write(configfile)


def ts():
    return int((
        datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)
    ).total_seconds())


def persist_nokia_refresh_token(token_dict, cfg_file):
    exp_time = str(ts()+int(token_dict['expires_in']))
    parser = configparser.ConfigParser()
    parser.read(cfg_file)
    parser.set('nokia', 'access_token', token_dict['access_token'])
    parser.set('nokia', 'refresh_token', token_dict['refresh_token'])
    parser.set('nokia', 'token_type', token_dict['token_type'])
    parser.set('nokia', 'token_expiry', exp_time)
    if 'user_id' in token_dict:
        parser.set('nokia', 'user_id', token_dict['userid'])
    with open(cfg_file, 'w') as configfile:
        parser.write(configfile)


def refresh_steps(cfg_file, engine, db_df):
    print("REFRESHING STEPS...")
    parser = configparser.ConfigParser()
    parser.read(cfg_file)
    consumer_key = parser.get('fitbit', 'consumer_key')
    consumer_secret = parser.get('fitbit', 'consumer_secret')
    access_token = parser.get('fitbit', 'access_token')
    refresh_token = parser.get('fitbit', 'refresh_token')
    expires_at = parser.get('fitbit', 'expires_at')

    auth_client = fitbit.Fitbit(consumer_key, consumer_secret,
                                access_token=access_token,
                                refresh_token=refresh_token,
                                expires_at=float(expires_at),
                                refresh_cb=(lambda x: persist_fitbit_refresh_token(x, cfg_file))
                                )

    [date_start, date_end] = get_target_date_endpoints('steps', db_df)
    steps = auth_client.time_series('activities/steps', base_date=date_start, end_date=date_end)
    date_values = [[pd.to_datetime(val['dateTime']), val['value']] for val in steps['activities-steps']]
    updated_df = insert_values(date_values, 'steps', db_df)
    updated_df[['steps']] = updated_df[['steps']].apply(pd.to_numeric)

    with engine.connect() as conn, conn.begin():
        updated_df.to_sql('fitness', conn, if_exists='replace')

    return updated_df


def get_mfp_calories(day_entry):
    if not day_entry:
        return None
    if 'calories' in day_entry.totals:
        return day_entry.totals['calories']
    return None


def refresh_calories(engine, db_df):
    print("REFRESHING CALORIES...")
    [date_start, date_end] = get_target_date_endpoints('calories', db_df)
    date_query = date_start
    date_diff = date_end - date_query
    days = date_diff.days + 1

    client = myfitnesspal.Client('jamieinfinity')

    diary_dump = []
    for i in range(days):
        diary_data = client.get_date(date_query)
        diary_dump.append(diary_data)
        date_query = date_query + datetime.timedelta(days=1)

    date_values = [[pd.to_datetime(x.date.strftime('%Y-%m-%d')), get_mfp_calories(x)] for x in
                   diary_dump]

    updated_df = insert_values(date_values, 'calories', db_df)

    # these values are missing or corrupted on the web site / service
    updated_df.loc[updated_df.index=='2018-09-29', 'calories'] = 2668
    updated_df.loc[updated_df.index=='2019-10-30', 'calories'] = 2220
    updated_df.loc[updated_df.index=='2019-10-31', 'calories'] = 2008

    with engine.connect() as conn, conn.begin():
        updated_df.to_sql('fitness', conn, if_exists='replace')

    return updated_df


def refresh_weight(cfg_file, engine, db_df):
    print("REFRESHING WEIGHT...")
    parser = configparser.ConfigParser()
    parser.read(cfg_file)
    client_id = parser.get('nokia', 'client_id')
    client_secret = parser.get('nokia', 'client_secret')
    access_token = parser.get('nokia', 'access_token')
    token_expiry = parser.get('nokia', 'token_expiry')
    token_type = parser.get('nokia', 'token_type')
    refresh_token = parser.get('nokia', 'refresh_token')
    user_id = parser.get('nokia', 'user_id')

    creds = NokiaCredentials(access_token=access_token,
                             token_expiry=token_expiry,
                             token_type=token_type,
                             refresh_token=refresh_token,
                             user_id=user_id,
                             client_id=client_id,
                             consumer_secret=client_secret)
    client = NokiaApi(creds, refresh_cb=(lambda x: persist_nokia_refresh_token(x, cfg_file)))

    [date_start, date_end] = get_target_date_endpoints('weight', db_df)
    date_query = date_start
    date_diff = date_end - date_query
    days = date_diff.days + 2

    measures = client.get_measures(meastype=1, limit=days)
    weight_json = [{'weight': (float("{:.1f}".format(x.weight * 2.20462))), 'date': x.date.strftime('%Y-%m-%d')} for x
                   in measures]
    date_values = [[pd.to_datetime(x['date']), x['weight']] for x in weight_json]
    date_values_imp = [[pd.to_datetime(x['date']), np.nan] for x in weight_json]
    updated_df = insert_values(date_values, 'weight', db_df)
    updated_df = insert_values(date_values_imp, 'weight_imputed', updated_df)

    with engine.connect() as conn, conn.begin():
        updated_df.to_sql('fitness', conn, if_exists='replace')

    return updated_df


def impute_missing_weights(engine, db_df):
    print("IMPUTING MISSING WEIGHTS...")

    db_df_copy = db_df.copy()
    db_df_copy.loc[db_df_copy.weight.isna() & db_df_copy.weight_imputed.isna(), 'weight_imputed'] = 1.0
    db_df_copy.loc[~db_df_copy.weight.isna() & db_df_copy.weight_imputed.isna(), 'weight_imputed'] = 0.0
    db_df_copy.loc[db_df_copy.weight_imputed == 1.0, 'weight'] = np.nan
    db_df_copy.weight.interpolate(limit=2, method='spline', order=2, inplace=True, limit_direction='both')

    db_df_copy['weight_measured'] = db_df_copy.weight
    db_df_copy.loc[db_df_copy.weight_imputed==1, 'weight_measured'] = np.nan

    with engine.connect() as conn, conn.begin():
        db_df_copy.to_sql('fitness', conn, if_exists='replace')
    return db_df_copy


def add_roll_avg_columns(engine, db_df):
    print("ADDING ROLLING AVG COLUMNS...")

    period = '7D'
    period_days = 7
    min_periods = 3

    data_df = db_df[['weight', 'calories', 'steps']].copy().rename(
        {'weight': 'w_7day_avg', 'calories': 'c_7day_avg', 'steps': 's_7day_avg'}, axis=1)
    data_df = data_df.rolling(period, min_periods=min_periods).mean()
    temp_df = pd.DataFrame(index=pd.date_range(start="2015-09-16", end=datetime.date.today()))
    data_df = pd.merge(temp_df, data_df, how='left', left_index=True, right_index=True)
    data_df.w_7day_avg.interpolate(limit=2, method='linear', inplace=True, limit_direction='both')
    data_df['w_7day_avg_last_week'] = data_df.w_7day_avg.shift(period_days)
    data_df['c_7day_avg_last_week'] = data_df.c_7day_avg.shift(period_days)
    data_df['s_7day_avg_last_week'] = data_df.s_7day_avg.shift(period_days)
    # data_df.dropna(inplace=True)
    data_df['w_7day_avg_weekly_diff'] = data_df['w_7day_avg'] - data_df['w_7day_avg_last_week']
    data_df = pd.merge(db_df[['weight', 'calories', 'steps', 'weight_imputed', 'weight_measured']].copy(), data_df, how='left', left_index=True, right_index=True)

    with engine.connect() as conn, conn.begin():
        data_df.to_sql('fitness', conn, if_exists='replace')
    return data_df

#########################################
##### WEIGHT FORECAST MODEL COLUMNS #####
#########################################

# Model v0.1
c_w = 0.9842664081035283
c_c = 0.001965638199353011
c_s = -4.621900527451458e-05
c_0 = -1.2110620297640367
alpha_s = -c_s/c_c
alpha_0 = -c_0/c_c
alpha_w = (1-c_w)/c_c
gamma = -np.log(c_w)


def w_next_week(w_curr_week, c_next_week, s_next_week):
    return c_0 + c_w*w_curr_week + c_c*c_next_week + c_s*s_next_week


def w_steady_state(c, s):
    return (c - alpha_s*s - alpha_0)/alpha_w


def c_steady_state(w, s):
    return alpha_s*s + alpha_0 + alpha_w*w


def w_forecast(t_weeks, w_curr_week, c, s):
    wss = w_steady_state(c, s)
    return (w_curr_week - wss)*np.exp(-gamma*t_weeks) + wss


def w_velocity(w_curr_week, c, s):
    wss = w_steady_state(c, s)
    return gamma*(wss - w_curr_week)


def add_weight_forecast_columns(engine, db_df):
    print("ADDING WEIGHT FORECAST COLUMNS...")

    updated_df = db_df.copy()
    updated_df['Mv1_0_weight_velocity'] = updated_df.apply(lambda row: w_velocity(row.w_7day_avg, row.c_7day_avg, row.s_7day_avg), axis=1)
    updated_df['Mv1_0_proj_weight_1mo'] = updated_df.apply(lambda row: w_forecast(4.3, row.w_7day_avg, row.c_7day_avg, row.s_7day_avg), axis=1)
    updated_df['Mv1_0_proj_weight_2mo'] = updated_df.apply(lambda row: w_forecast(8.6, row.w_7day_avg, row.c_7day_avg, row.s_7day_avg), axis=1)

    with engine.connect() as conn, conn.begin():
        updated_df.to_sql('fitness', conn, if_exists='replace')

    return updated_df

























