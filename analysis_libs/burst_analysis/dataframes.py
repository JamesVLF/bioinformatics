import pandas as pd


class DataFrameManager:
    def __init__(self):
        self.frames = {}

    def add(self, name, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.frames[name] = df

    def get(self, name):
        return self.frames.get(name)

    def append(self, name, records, overwrite=False):
        new = pd.DataFrame(records)
        if overwrite or name not in self.frames:
            self.frames[name] = new
        else:
            self.frames[name] = pd.concat([self.frames[name], new], ignore_index=True)

    def update_column(self, name, column, values):
        if name not in self.frames:
            raise KeyError(f"DataFrame '{name}' not found")
        self.frames[name][column] = values

    def drop(self, name):
        self.frames.pop(name, None)

    def list(self):
        return list(self.frames.keys())
