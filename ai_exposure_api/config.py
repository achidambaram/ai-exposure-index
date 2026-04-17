from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

ONET_ABILITIES_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Abilities.txt"
ONET_OCCUPATIONS_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Occupation%20Data.txt"
ONET_TASKS_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Task%20Statements.txt"
ONET_TASK_RATINGS_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Task%20Ratings.txt"
ONET_TASKS_TO_DWAS_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Tasks%20to%20DWAs.txt"
AIOE_URL = "https://github.com/AIOE-Data/AIOE/raw/refs/heads/main/AIOE_DataAppendix.xlsx"

ABILITIES_FILE = DATA_DIR / "Abilities.txt"
OCCUPATIONS_FILE = DATA_DIR / "Occupation Data.txt"
TASKS_FILE = DATA_DIR / "Task Statements.txt"
TASK_RATINGS_FILE = DATA_DIR / "Task Ratings.txt"
TASKS_TO_DWAS_FILE = DATA_DIR / "Tasks to DWAs.txt"
AIOE_FILE = DATA_DIR / "AIOE_DataAppendix.xlsx"
TRAINING_FILE = DATA_DIR / "training_dataset.csv"
PREDICTIONS_FILE = DATA_DIR / "model_predictions.csv"
OCCUPATION_LOOKUP_FILE = DATA_DIR / "occupation_lookup.csv"
ABILITY_CONTRIBUTIONS_FILE = DATA_DIR / "ability_contributions.csv"
METRICS_FILE = MODEL_DIR / "metrics.json"
MODEL_FILE = MODEL_DIR / "best_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"
FEATURES_FILE = MODEL_DIR / "feature_columns.json"
TRAIN_FRAME_FILE = MODEL_DIR / "train_frame.csv"
