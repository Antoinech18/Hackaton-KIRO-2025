import pandas as pd
import numpy as np

data = pd.read_csv("sujet/instances/vehicles.csv")

columns = data.columns[1:]
# ['max_capacity', 'rental_cost', 'fuel_cost', 'radius_cost', 'speed',
#       'parking_time', 'fourier_cos_0', 'fourier_sin_0', 'fourier_cos_1',
#       'fourier_sin_1', 'fourier_cos_2', 'fourier_sin_2', 'fourier_cos_3',
#       'fourier_sin_3']

vehicles1 = {}
vehicles2 = {}
vehicles3 = {}

for c in columns: 
    vehicles1[c] = data.loc[0][c]
    vehicles2[c] = data.loc[1][c]
    vehicles3[c] = data.loc[2][c]


