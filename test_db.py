from src.batch import run_batch_extraction
from src.database import save_features, get_db_summary

df = run_batch_extraction("data/")
result = save_features(df)
print(result)
print(get_db_summary())