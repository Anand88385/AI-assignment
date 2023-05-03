import requests
import io
import pandas as pd


url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjK3-3t20y0_yukyGbdKvtHfVb0PAgBckE82ZzrlLQX9cFGrLhC_9ejHsmlrTMyTgAl52rEovdzuyG/pub?gid=0&single=true&output=csv'

response = requests.get(url)


csv_str = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(csv_str)

print(df.head())
