import pandas as pd

def load_data(reference_file, current_file):

    reference = pd.read_csv(reference_file)
    current = pd.read_csv(current_file)

    return reference, current