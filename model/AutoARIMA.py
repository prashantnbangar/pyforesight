import json
import os

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'parameters.json')) as f:
    parameters = json.load(f)

class AutoARIMA():
    def __init__(self) -> None:
        super().__init__()
        print(parameters)

    def fit(self, dataframe):
        print(dataframe.head(5))