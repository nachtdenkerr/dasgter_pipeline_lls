import os
import shutil
import numpy as np
import pandas as pd
import dagster as dg
from dagster import AssetKey

DATASET_DIR = "data/incoming"

#@dg.asset(config_schema={"batch_name": str})
#def load_batch(context):
#    batch_name = context.op_config["batch_name"]
#    batch_dir = os.path.join(DATASET_DIR, batch_name)

#    data_frames = []

#    for f in os.listdir(batch_dir):
#        if f.endswith(".csv"):
#            df = pd.read_csv(os.path.join(batch_dir, f))
#            data_frames.append(df)

#    context.log.info(f"Loaded {len(data_frames)} CSVs from batch {batch_name}")
    
#    # Return a merged dataframe or a dict of per-vehicle dfs
#    return pd.concat(data_frames, ignore_index=True)


@dg.asset(
	group_name='preprocess',
	config_schema={"batch_name": str},   # <-- this is the correct way for assets
)
def archive_batch(context: dg.AssetExecutionContext, raw_data):
    batch_name = context.op_config["batch_name"]
    src = f"data/incoming/{batch_name}"
    dst = f"data/processed/{batch_name}"
    shutil.move(src, dst)
    context.log.info(f"Moved {batch_name} to processed/")

HEADERS = ['FFID', 'Height', 'Loaded', 'OnDuty', 'TimeStamp', 'Latitude', 'Longitude', 'Speed']


# MARK:read csv
@dg.asset(
	io_manager_key="parquet_io_manager",
	key=AssetKey(["raw_data"]),
	group_name='preprocess',
	config_schema={"batch_name": str}
)
def s1_read_csv(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
	"""
	Read .csv file from DATASET_DIR
	"""
	batch_name = context.op_config["batch_name"]
	batch_dir = os.path.join(DATASET_DIR, batch_name)

	context.log.info(f"Reading CSV files from batch: {batch_name}")

	dataframes = []
	for file in os.listdir(batch_dir):
		if file.endswith(".csv"):
			file_path = os.path.join(batch_dir, file)
			# Read CSV and assign headers
			df = pd.read_csv(file_path, names=HEADERS, header=None, sep=";")
			# Use filename (without extension) as key
			dataframes.append(df)
	df = pd.concat(dataframes, ignore_index=True)

	context.log.info(f"Read {len(dataframes)} CSV files from batch {batch_name}..")
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")
	return df


# MARK:add vehicle types
@dg.asset(
	io_manager_key="parquet_io_manager",
	key=AssetKey('df_with_types'),
	required_resource_keys={"vehicle_config"},
	group_name='preprocess',

)
def s2_add_vehicle_types(context: dg.AssetExecutionContext, raw_data: pd.DataFrame):
	"""
	Adding vehicle types to the df: Towtruck, DieselForklift, ElectroForklift
	Processing the data to optimize memory usage
	"""

	context.log.info("Adding vehicle types...")
	df = raw_data
	vehicles = context.resources.vehicle_config

	# Create a mapping dictionary for all FFIDs
	type_map = {}
	type_map.update({ffid: "Towtruck" for ffid in vehicles['towtruck']})
	type_map.update({ffid: "DieselForklift" for ffid in vehicles['dieselForklift']})
	type_map.update({ffid: "ElectroForklift" for ffid in vehicles['electroForklift']})
	
	# Assign "Type" column by mapping FFID → category
	df["Type"] = df["FFID"].map(type_map)
	df['Loaded'] = df['Loaded'].astype('int8')
	df['Speed'] = df['Speed'].astype('float32')
	df['Height'] = df['Height'].astype('float32')
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Unique types: {df['Type'].unique()}")
	context.log.info(f"Column names: {df.columns.to_list()}")
	return df


# MARK:filter
TIME_BIN = 5  # in minutes
@dg.asset(
	io_manager_key='parquet_io_manager',
	key=AssetKey("filtered_data"),
	required_resource_keys={"vehicle_config"},
	group_name='preprocess',

)
def s3_filter(context: dg.AssetExecutionContext, df_with_types: pd.DataFrame) -> dg.MaterializeResult:
	"""
	Filter out the indoor vehicles, anomalies, add timebin
	"""

	context.log.info("Filtering on-duty outdoor vehicles...")
	vehicles = context.resources.vehicle_config
	indoor= vehicles['indoor']

	df = df_with_types
	df = df[~(df['FFID'].isin(indoor)) & (df['OnDuty'] == 1)].reset_index(drop=True)
	df.drop(columns=['OnDuty'], axis=1)
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")

	context.log.info("Filtering height > 7 and speed > 30 anomalies...")
	df = df[(df['Height'] <= 7) & (df['Speed'] <= 30)].copy()
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")

	df['Datetime'] = pd.to_datetime(df['TimeStamp'], errors='coerce', unit='ms') + pd.Timedelta(hours=2)
	df.drop(columns=['TimeStamp'], axis=1)
	df['Weekday'] = df['Datetime'].dt.weekday
	df['Date'] = df['Datetime'].dt.date
	df['Timebin'] = df['Datetime'].dt.floor(f'{TIME_BIN}min')
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")
	return df


# MARK:select active rows
@dg.asset(
    io_manager_key='parquet_io_manager',
    key=AssetKey("df_active_forklifts"),
	group_name='preprocess',

)
def s4_select_active(context: dg.AssetExecutionContext, filtered_data: pd.DataFrame) -> dg.MaterializeResult:
	"""
	Select active vehicles: handling cargo, driving goods or idle state shorter than 5 mins
	"""
	context.log.info("Selecting active rows...")

	df = filtered_data.copy()
	context.log.info(f"DataFrame shape: {df.shape}")
	context.log.info(f"Columns: {df.columns.tolist()}")
	if df.empty:
		context.log.warn("DataFrame is empty!")
		return dg.MaterializeResult(value=df, metadata={"rows": 0, "columns": 0})

	# Sort by equipment and time
	df.sort_values(['FFID', 'Datetime'], inplace=True)

	# Extract numpy arrays for efficient operations
	ffid = df['FFID'].to_numpy()
	speed = df['Speed'].to_numpy()
	loaded = df['Loaded'].to_numpy()
	height = df['Height'].to_numpy()
	dt = df['Datetime'].to_numpy()

	# Compute active moving (speed >1 and loaded)
	moving_with_load = (speed > 1) & (loaded == 1)

	# Compute time difference per equipment
	time_diff = np.diff(dt.astype('datetime64[s]'), prepend=dt[0])
	time_diff_seconds = time_diff / np.timedelta64(1, 's')  # convert to float seconds

	same_ffid = np.concatenate([[False], ffid[1:] == ffid[:-1]])
	time_2s_prev = same_ffid & (time_diff_seconds >= 1.5) & (time_diff_seconds <= 4)

	# Cargo handling: height or load change
	height_change = np.concatenate([[False], ~np.isclose(height[1:], height[:-1], equal_nan=True)])
	load_change = np.concatenate([[False], ~np.isclose(loaded[1:], loaded[:-1], equal_nan=True)])
	cargo_handling = time_2s_prev & (height_change | load_change)

	# Active mask
	active_mask = moving_with_load | cargo_handling
	idle_mask = ~active_mask

	# Compute idle duration per row using groupby transform (vectorized)
	df['Idle_Group'] = (idle_mask != np.roll(idle_mask, 1)).cumsum()
	idle_durations = (
		df[idle_mask]
		.groupby(['FFID', 'Idle_Group'])['Datetime']
		.transform(lambda x: (x.max() - x.min()).total_seconds() / 60)
	)
	# Mark rows to remove if idle > 5 minutes
	to_remove = idle_mask.copy()
	to_remove[idle_mask] = idle_durations > 5

	# Filter active rows
	df_active = df[~to_remove].copy()
	df_active.drop(columns=['Idle_Group'], inplace=True)

	context.log.info(f"Active rows after filtering: {df_active.shape[0]}")
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")
	return df
