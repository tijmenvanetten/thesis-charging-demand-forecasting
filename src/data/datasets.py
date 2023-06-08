from data import load_covariates, load_target
# from ...config import SHELL_DATASET_PATH, PALO_ALTO_DATASET_PATH, BOULDER_DATASET_PATH, SHELL_WEEKLY_DATASET_PATH

def data_handler(dataset:str, use_static_cols=None):
    if dataset == 'palo_alto':
        return {
            "target": load_target(
                    path='C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/palo_alto_dataset.csv',
                    time_col='date',
                    static_cols=None,
                    freq='D'
                ),
        }
    elif dataset == 'shell':
        return {
            "target": load_target(
                    path='C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset.csv',
                    time_col='date',
                    static_cols=None,
                    freq='D'
                ),
            "covariates":  load_covariates(
                    path='data/03_processed/weather_ecad.csv',
                    value_cols=['temp_max', 'temp_min', 'sunshine', 'precip'],
                    freq='D'
                )
        }
        
    elif dataset == 'boulder':
        return {
            "target": load_target(
                    path='C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/boulder_dataset.csv',
                    time_col='date',
                    static_cols=None,
                    freq='D'
                ),
        }
    elif dataset == 'shell_weekly':
        return {
            "target": load_target(
                    path='C:/Users/tijmen.vanetten/Documents/emobility-vanetten/data/03_processed/shell_dataset_weekly.csv',
                    time_col='date',
                    static_cols=None,
                    freq='W-MON'
                ),
        }
    raise "Dataset not found"