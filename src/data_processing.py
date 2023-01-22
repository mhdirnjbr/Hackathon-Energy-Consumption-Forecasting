import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

def clean_features(orig_df):
    df = orig_df.copy()
    
    df = drop_features(df)

    df = median_filler(df)

    df = mean_filler(df)

    df = mode_filler(df)

    df = one_hot_encode(df)

    df = one_hot_list_columns(df)

    df = one_hot_symbol_columns(df)
    
    df = process_lower_floor_adjacency_type(df)
    df = process_lower_floor_insulation_type(df)
    df = process_lower_floor_material(df)

    df = process_cat_pct(df, 'main_heat_generators')
    df = process_cat_pct(df, 'main_water_heaters')

    df['years_old'] = 2023 - df['building_year']
    df['nb_dwellings'] = df['nb_dwellings'].clip(upper=50)
    df['outer_wall_thickness'] = df['outer_wall_thickness'].apply(lambda x: float(re.sub(' et -', '', x)))


    list_to_group = [
            "outer_wall_materials",
            "upper_floor_material",
            "window_frame_material",
            "additional_heat_generators"
        ]


    list_to_binarize = [
        "water_heating_type",
        "heating_type",
    ]
    
    for col in list_to_binarize:
        df[col] = df[col].fillna(df[col].mode())
        df[col] = df[col].replace({df[col].unique()[0]: 0, df[col].unique()[1]: 1})

    
    for col in list_to_group:
        df = re_categorize_by_count(df, col, 1000)

    df = encode_levels(df, 'radon_risk_level')
    df = encode_levels(df, 'clay_risk_level')
    df = encode_levels(df, 'thermal_inertia')

    df['renewable_energy_sources'] = df['renewable_energy_sources'].fillna('')
    df['solar thermal (ecs)'] = df['renewable_energy_sources'].str.contains('ecs')
    df['solar photovoltaic'] = df['renewable_energy_sources'].str.contains('solar')
    df['solar thermal (heating)'] = df['renewable_energy_sources'].str.contains('heating')
    df['solar thermal (hot water)'] = df['renewable_energy_sources'].str.contains('hot')
    df['solar thermal (DHW)'] = df['renewable_energy_sources'].str.contains('DHW')
    df.drop(columns='renewable_energy_sources', inplace=True)
    

    df['upper_floor_insulation_type'] = df['upper_floor_insulation_type'].fillna('INTERNAL')
    df['upper_floor_insulation_typeINTERNAL'] = df['upper_floor_insulation_type'].str.contains('INTERNAL')
    df['upper_floor_insulation_typeUNINSULATED'] = df['upper_floor_insulation_type'].str.contains('UNINSULATED')
    df['upper_floor_insulation_typeEXTERNAL'] = df['upper_floor_insulation_type'].str.contains('EXTERNAL')
    df['upper_floor_insulation_typeREFLEXION'] = df['upper_floor_insulation_type'].str.contains('REFLEXION')
    df.drop(columns='upper_floor_insulation_type', inplace=True)

    
    df['wall_insulation_type'] = df['wall_insulation_type'].fillna('internal')
    df['wall_insulation_type_internal'] = df['wall_insulation_type'].str.contains('internal')
    df['wall_insulation_type non insulated'] = df['wall_insulation_type'].str.contains('non insulated')
    df['wall_insulation_type external'] = df['wall_insulation_type'].str.contains('external')
    df['wall_insulation_type insulated'] = df['wall_insulation_type'].str.contains('insulated')
    df['wall_insulation_type reflection'] = df['wall_insulation_type'].str.contains('(?:reflection|reflexion)', regex=True)
    df.drop(columns='wall_insulation_type', inplace=True)   

    df['volumn'] = df['building_height_ft'] * df['building_total_area_sqft'] 
    
    return df


def drop_features(df):
    DROP_COLUMNS = [
        'main_heating_type',
        'main_water_heating_type',
        'nb_commercial_units',
        'nb_gas_meters_commercial',
        'nb_gas_meters_housing',
        'nb_gas_meters_total',
        'nb_housing_units',
        'nb_meters',
        'nb_parking_spaces',
        'nb_power_meters_commercial',
        'building_period',
        'building_use_type_description',
        'nb_power_meters_housing',
        'nb_power_meters_total',
        'nb_units_total',
        'consumption_measurement_date',
        "balcony_depth"
    ]

    df = df.drop(columns=DROP_COLUMNS)
    return df


def median_filler(df):
    MEDIAN_FEATURES = [
        "building_year",
        "outer_wall_thermal_conductivity",
        "window_heat_retention_factor",
        "window_thermal_conductivity",
        "building_height_ft",
        "living_area_sqft",
        "building_total_area_sqft",
        "upper_floor_thermal_conductivity",
        "lowe_floor_thermal_conductivity"
    ]

    for feature in MEDIAN_FEATURES:
        df[feature] = df[feature].fillna(df[feature].median())

    return df

def mean_filler(df):
    MEAN_FEATURES = [
        "percentage_glazed_surfaced",
        "altitude"
    ]

    for feature in MEAN_FEATURES:
        df[feature] = df[feature].fillna(df[feature].median())

    return df

def mode_filler(df):
    MODE_FEATURES = [
        "radon_risk_level",
        "outer_wall_thickness",
        "clay_risk_level",
        "has_balcony"
    ]

    for feature in MODE_FEATURES:
        df[feature] = df[feature].fillna(df[feature].mode().iloc[0])

    return df
    
def one_hot_encode(df):
    ONE_HOT_COLS = [
        'is_crossing_building',
        'roof_material',
        'upper_floor_adjacency_type',
        'window_glazing_type',
        'ventilation_type',
        "building_type",
        "building_use_type_code",
        "additional_water_heaters", #many NaNs
        "bearing_wall_material",
        "building_category",
        "building_class",
        "window_filling_type"
    ]

    for column in ONE_HOT_COLS:
        df = create_one_hot(df, column)

    return df


def one_hot_list_columns(df):
    OH_LIST_COLS = [
        'heat_generators',
        'water_heaters',
        'window_orientation'
    ]
    for column in OH_LIST_COLS:
        df = create_one_hot_ColumnOfLists(df, column)

    return df


def one_hot_symbol_columns(df):
    OH_SYMBOL_COLS = {
        "heating_energy_source": ' + ',
        "water_heating_energy_source": ' + ',
    }

    for column, symbol in OH_SYMBOL_COLS.items():
        df = create_one_hot_ColumnSplitBySymbol(df, column, symbol)

    return df


def encode_levels(df, column):
    LEVEL_DICT = {
        'low': 0, 
        'medium': 1, 
        'high': 2, 
        'very high': 3
    }

    df[column] = df[column].replace(LEVEL_DICT)
    return df


def process_lower_floor_insulation_type(df):
    np.random.seed(0)
    insulation_df = pd.DataFrame()
    insulation_df['external_insulation'] = df['lower_floor_insulation_type'].str.contains('external').fillna(False).astype(int)
    insulation_df['internal_insulation'] = df['lower_floor_insulation_type'].str.contains('internal').fillna(False).astype(int)
    
    undetermined_insulation_mask = df['lower_floor_insulation_type'] == 'insulated'
    insulation_df[undetermined_insulation_mask] = np.random.randint(low=0, high=2, size=(np.sum(undetermined_insulation_mask), 2))
    
    df = pd.concat([df.drop(columns=['lower_floor_insulation_type']), insulation_df], axis=1)
    
    return df

def process_lower_floor_material(df):
    floor_df = pd.DataFrame()
    floor_df['concrete_slab_floor'] = (df['lower_floor_material'] == 'concrete slab')
    floor_df['heavy_floor'] = (df['lower_floor_material'] == 'heavy floor, such as clay floor joists, concrete beams')
    floor_df['insulated_joist_floor'] = (df['lower_floor_material'] == 'Insulated joist floor')
    floor_df['wood_floor'] = df['lower_floor_material'].str.contains('wood', case=False)
    floor_df['metal_floor'] = df['lower_floor_material'].str.contains('metal', case=False)
    
    floor_df['other_floor'] = ~floor_df.any(axis=1)
    
    floor_df = floor_df.fillna(False).astype(int)
    
    df = pd.concat([df.drop(columns=['lower_floor_material']), floor_df], axis=1)
    
    return df

def process_lower_floor_adjacency_type(df):
    np.random.seed(0)
    prob_series = df['lower_floor_adjacency_type'].value_counts(normalize=True)
    categories = prob_series.index.tolist()
    probabilities = prob_series.values.tolist()
   
    col_na_mask = df['lower_floor_adjacency_type'].isna()
    df['lower_floor_adjacency_type'][col_na_mask] = np.random.choice(categories, size=np.sum(col_na_mask), p=probabilities)
   
    one_hot_df = pd.get_dummies(df['lower_floor_adjacency_type'], prefix='lower_floor_adjacency_type', drop_first=True)
   
    df = pd.concat([df.drop(columns='lower_floor_adjacency_type'), one_hot_df], axis=1)
   
    return df

def process_cat_pct(df, column, new_col_name=None):
    cat_data_pct = df[column].value_counts().cumsum()/len(df)
    keep_cats = cat_data_pct[cat_data_pct<0.95].index.tolist()
    
    if cat_data_pct.iloc[0] > 0.3:
        df[column] = df[column].fillna(cat_data_pct.index[0])
        
    if not new_col_name:
        new_col_name = column
    
    for category in keep_cats:
        df[f'{new_col_name}_{category}'] = (df[column] == category).astype(int)
        
    df.drop(columns=[column], inplace = True)
    
    return df

def create_one_hot_ColumnSplitBySymbol(df, col_name, symbol): #symbol = ' + '
    df[col_name] = df[col_name].fillna('none')
    df[col_name] = df[col_name].apply(lambda x: 'none' if x=='' else x)
    new_df = pd.get_dummies(pd.DataFrame(df[col_name].apply(lambda x: x.split(symbol)).tolist()))
    new_df.columns = new_df.columns.str.split("_").str[-1]

    merged_df = pd.DataFrame()
    for unique_column in new_df.columns.unique():
        if len(new_df[unique_column].shape) == 1:
            merged_df[f'{col_name}_{unique_column}_merged'] = new_df[unique_column]
        else:
            merged_df[f'{col_name}_{unique_column}_merged'] = np.zeros(new_df[unique_column].shape[0], dtype=np.int32)
            for icol in range(new_df[unique_column].shape[1]):
                merged_df[f'{col_name}_{unique_column}_merged'] = merged_df[f'{col_name}_{unique_column}_merged'] | new_df[unique_column].iloc[:,icol]
    df = pd.concat([df, merged_df], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df

def create_one_hot_ColumnOfLists(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mode().iloc[0])
    df[col_name] = df[col_name].apply(lambda x: '[empty]' if x=='[]' else x)
    new_df = pd.get_dummies(pd.DataFrame(df[col_name].apply(lambda x: x[1:-1].split(',')).tolist()))
    new_df.columns = new_df.columns.str.split("_").str[-1]

    merged_df = pd.DataFrame()
    for unique_column in new_df.columns.unique():
        if len(new_df[unique_column].shape) == 1:
            merged_df[f'{col_name}_{unique_column}_merged'] = new_df[unique_column]
        else:
            merged_df[f'{col_name}_{unique_column}_merged'] = np.zeros(new_df[unique_column].shape[0], dtype=np.int32)
            for icol in range(new_df[unique_column].shape[1]):
                merged_df[f'{col_name}_{unique_column}_merged'] = merged_df[f'{col_name}_{unique_column}_merged'] | new_df[unique_column].iloc[:,icol]
    df = pd.concat([df, merged_df], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df

def create_one_hot(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mode().iloc[0])
    df = pd.concat([df, pd.get_dummies(df[col_name])], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df

def re_categorize_by_count(df, col_name, threshold):
    df[col_name] = df[col_name].fillna(df[col_name].mode().iloc[0])
    df[col_name] = df[col_name].apply(lambda x: '[empty]' if x=='[]' else x)
    
    dict_col = df[col_name].value_counts().to_dict()
    df[col_name] = df[col_name].apply(lambda x: '[other]' if dict_col[x]<threshold else x)
    
    new_df = pd.get_dummies(pd.DataFrame(df[col_name].apply(lambda x: x[1:-1].split(',')).tolist()))
    new_df.columns = new_df.columns.str.split("_").str[-1]

    merged_df = pd.DataFrame()
    for unique_column in new_df.columns.unique():
        if len(new_df[unique_column].shape) == 1:
            merged_df[f'{col_name}_{unique_column}_merged'] = new_df[unique_column]
        else:
            merged_df[f'{col_name}_{unique_column}_merged'] = np.zeros(new_df[unique_column].shape[0], dtype=np.int32)
            for icol in range(new_df[unique_column].shape[1]):
                merged_df[f'{col_name}_{unique_column}_merged'] = merged_df[f'{col_name}_{unique_column}_merged'] | new_df[unique_column].iloc[:,icol]
    df = pd.concat([df, merged_df], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df



if __name__ == "__main__":
    df = pd.read_csv('~/hfactory_magic_folders/hi__paris_hackathon/building_energy_efficiency/datasets/train/train_features_sent.csv', nrows=50000) ## UNCOMMENT FOR FULL DATA
    test_df = pd.read_csv('~/hfactory_magic_folders/hi__paris_hackathon/building_energy_efficiency/datasets/test/test_features_sent.csv', nrows=50000) ## UNCOMMENT FOR FULL DATA

    df = clean_features(df)
    test_df = clean_features(test_df)

#     df.to_csv('../data/train.csv')
#     test_df.to_csv('../data/test.csv')