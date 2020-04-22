#!/Users/jamieinfinity/opt/anaconda3/envs/WeightForecaster/bin/python
import sys
toolpath = '/Users/jamieinfinity/Dropbox/Projects/WeightForecaster/weightforecaster/server/src'
sys.path.append(toolpath)
from wtfc_utils import etl_utils as etl
import os
import datetime
from sqlalchemy import create_engine
import pandas as pd


server_dir = '/Users/jamieinfinity/Dropbox/Projects/WeightForecaster/weightforecaster/server/'
cfg_file = server_dir + 'config/api_params.cfg'
db_dir = server_dir + 'db/'
backups_dir = db_dir + 'backups/'
db_name = 'weightforecaster'
db_ext = '.db'
db_file_name = db_dir + db_name + db_ext


engine = create_engine('sqlite:///'+db_file_name)
with engine.connect() as conn, conn.begin():
    db_df = pd.read_sql_table('fitness', conn, index_col='date', parse_dates=['date'])


if os.path.isfile(db_file_name):
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    backup_file_name = backups_dir + db_name + '_BACKUP_' + timestamp + db_ext
    etl.copy_file(db_file_name, backup_file_name)


updated_df = etl.refresh_steps(cfg_file, engine, db_df)
updated_df = etl.refresh_weight(cfg_file, engine, updated_df)
updated_df = etl.refresh_calories(engine, updated_df)
updated_df = etl.impute_missing_weights(engine, updated_df)
updated_df = etl.add_roll_avg_columns(engine, updated_df)
