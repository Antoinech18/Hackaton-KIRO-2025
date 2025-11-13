import pandas as pd
import numpy as np

##### Extraction des données #####

data_vehicles = pd.read_csv("sujet/instances/vehicles.csv")

columns = data_vehicles.columns[1:]
# ['max_capacity', 'rental_cost', 'fuel_cost', 'radius_cost', 'speed',
#       'parking_time', 'fourier_cos_0', 'fourier_sin_0', 'fourier_cos_1',
#       'fourier_sin_1', 'fourier_cos_2', 'fourier_sin_2', 'fourier_cos_3',
#       'fourier_sin_3']

vehicles1 = {}
vehicles2 = {}
vehicles3 = {}

for c in columns: 
    vehicles1[c] = data_vehicles.loc[0][c]
    vehicles2[c] = data_vehicles.loc[1][c]
    vehicles3[c] = data_vehicles.loc[2][c]

vehicles = [vehicles1, vehicles2, vehicles3]

import math
import numpy as np
def yjminyi(phij,phii):
    return 6.371E6*2*np.pi*(phij-phii)/360
def xjminxi(lambdaj,lambdai):
    return 6.371E6*math.cos(2*np.pi*48.764246/360)*2*np.pi*(lambdaj-lambdai)/360
