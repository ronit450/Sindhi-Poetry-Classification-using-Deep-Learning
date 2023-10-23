import pandas as pd

df = pd.read_csv('dataBhitai.csv')
a = df.head()
print(a['Couplets'][0])
