import os
import glob
import pandas as pd


def load_spark_output_csv(folder_path: str) -> pd.DataFrame:
    """
    Load a single-part Spark CSV output (part-*.csv) into a pandas DataFrame.
    """
    part_files = glob.glob(os.path.join(folder_path, "part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No part-*.csv found in {folder_path}")
    return pd.read_csv(part_files[0])
