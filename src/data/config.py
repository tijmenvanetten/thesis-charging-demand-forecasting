from pathlib import Path 

SESSION_DATA_SRC_DIR = 'data/raw/sessions'
SESSION_DATA_TARGET_DIR = 'data/processed'
SESSION_DATA_TARGET_NAME = 'sessions_data.csv'
LOCATION_TYPES = ['On Forecourt', 'Mobility Hub', 'Destination']

# per charger
# supply bottleneck
# range anxiety
# clustering


SESSION_DATA_DIR = Path(SESSION_DATA_TARGET_DIR) / Path(SESSION_DATA_TARGET_NAME)