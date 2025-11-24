import numpy as np
import pandas as pd
import dagster as dg
from dagster import AssetKey
from shapely.geometry import Point


# MARK:compute vehicles in area
@dg.asset(
	io_manager_key='parquet_io_manager',
	key=AssetKey(["df_num_forklifts"]),
	required_resource_keys={"area_config"},
	group_name='compute',
)
def s5_compute_avg_vehicles_in_area(context: dg.AssetExecutionContext, df_active_forklifts: pd.DataFrame) -> pd.DataFrame:
	"""
	Computing number of vehicles in predefined areas at time of day
	"""
	context.log.info("Computing average vehicles in predefined areas...")
	results = []
	df = df_active_forklifts

	areas = context.resources.area_config 
	for area_name, poly in areas.items():
		in_area = df[
			df.apply(lambda r: poly.contains(Point(r["Latitude"], r["Longitude"])), axis=1)
		].copy()

		if len(in_area) == 0:
			continue  # skip if no data in this area

		agg_per_5min = (
			in_area.groupby(["Date", "Timebin", "Weekday", "Type"])
			.agg(count=("FFID", "nunique"))
			.reset_index()
		)

		pivoted = agg_per_5min.pivot_table(
			index=["Date", "Timebin", "Weekday"],
			columns="Type",
			values="count",
			fill_value=0
		).reset_index()

		pivoted.columns.name = None

		pivoted['TotalForklifts'] = pivoted['DieselForklift'] + pivoted['ElectroForklift']
		pivoted["Area"] = area_name
		results.append(pivoted)
	df_group_by_area = pd.concat(results, ignore_index=True)
	df_group_by_area['Hour'] = df_group_by_area['Timebin'].dt.hour + (df_group_by_area['Timebin'].dt.minute / 60)
	context.log.info(f'{df_group_by_area["Area"].unique()}')

	context.log.info("Dataframe first 5 rows")
	context.log.info(df_group_by_area.head())
	context.log.info(f"Dataframe shape {df_group_by_area.shape}")
	context.log.info(f"Column names: {df_group_by_area.columns.to_list()}")
	return df_group_by_area


# MARK:pivot areas
@dg.asset(
	io_manager_key='parquet_io_manager',
	key=AssetKey('df_pivot_areas'),
	group_name='compute',

)
def s6_pivot_areas(context: dg.AssetExecutionContext, df_num_forklifts: pd.DataFrame) -> pd.DataFrame:
	context.log.info("Creating new columns based on area and vehicle types ...")

	df = df_num_forklifts
	df_hope = pd.DataFrame()
	context.log.info(f"Column names: {df.columns.to_list()}")

	for type in ["ElectroForklift", "DieselForklift", "TotalForklifts"]:
		pivot_df = df.pivot_table(
			index=['Timebin', 'Hour', 'Weekday'],
			columns='Area',
			values=type,
			aggfunc='first',  # Use first since we expect one value per timebin-area combo
			fill_value=0
		)
		# Rename columns to add suffix
		pivot_df.columns = [f'{col}_{type}' for col in pivot_df.columns]
		df_hope = pd.concat([df_hope, pivot_df], axis=1)

	# Reset index to make Timebin, Hour, Weekday regular columns
	df_hope = df_hope.reset_index()
	context.log.info(f"Dataframe shape {df_hope.shape}")
	context.log.info(f"Column names: {df_hope.columns.to_list()}")
	return df_hope

# MARK:create time sequence
@dg.asset(
	io_manager_key='parquet_io_manager',
	key=AssetKey('df_time_sequence'),
	group_name='compute',

)
def s7_fill_with_zeros(context: dg.AssetExecutionContext, df_pivot_areas: pd.DataFrame) -> pd.DataFrame:
	context.log.info("Creating time series with 0 at NaN values ...\n")

	df = df_pivot_areas
	df = df.set_index('Timebin')
	df.sort_index(inplace=True)

	# Get the absolute earliest and latest time stamps for Tor 2
	start_time = df.index.min()
	end_time = df.index.max()

	# Create a complete, continuous 5-minute index for the whole duration
	full_time_index = pd.date_range(start=start_time, end=end_time, freq='5Min')
	context.log.info(f"Start: {df.index.min()}  End: {df.index.max()}")

	# Identify the columns that should be filled with 0.0
	column_names = df.columns.to_list()
	non_fill_cols = ['Date', 'Timebin', 'Weekday', 'Area', 'Hour']
	fill_cols = [col for col in column_names if col not in non_fill_cols]
	
	# Reindex the DataFrame to the full timeline. This inserts NaN rows where gaps existed.
	df_filled = df.reindex(full_time_index)
	full_time_index = pd.date_range(start=start_time, end=end_time, freq='5min')
	context.log.info(f"unique full_time_index length: {len(pd.Index(full_time_index).unique())}")

	# Fill the numerical count columns (where gaps are) with 0.0
	df_filled[fill_cols] = df_filled[fill_cols].fillna(0.0)

	# Reset the index to make 'time_bin' a regular column again
	df_filled = df_filled.reset_index().rename(columns={'index': 'Timebin'})
	df_filled['Hour'] = df_filled['Timebin'].dt.hour
	context.log.info(f"Dataframe shape {df_filled.shape}")
	context.log.info(f"Column names: {df_filled.columns.to_list()}")
	context.log.info(f"Dataframe head \n{df_filled.head()}")
	return df_filled


@dg.asset(
	io_manager_key='parquet_io_manager',
	key=AssetKey('df'),
	group_name='compute',

)
def s8_feature_engineering(context: dg.AssetExecutionContext, df_time_sequence) -> pd.DataFrame:
	context.log.info("Converting hour and weekday to sine and cosine values ...")
	df = df_time_sequence
	df['Weekday'] = df['Timebin'].dt.weekday

	df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
	df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)

	df['WeekdaySin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
	df['WeekdayCos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
	context.log.info(f"Dataframe shape {df.shape}")
	context.log.info(f"Column names: {df.columns.to_list()}")
	df.to_csv("5min.csv")
	return df
