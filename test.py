from src.batch import run_batch_extraction
df = run_batch_extraction("data/")
print(df.shape)