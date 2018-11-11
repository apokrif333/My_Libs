import json
import pandas as pd

with open('C:/Users/Tom/PycharmProjects/Start/GibHub/My_Libs/test_data/train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)


