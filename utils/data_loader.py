import pandas as pd

def load_recipients(path):
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
