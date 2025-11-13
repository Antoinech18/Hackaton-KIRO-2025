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
import pandas as pd
import numpy as np
import math

##### Constantes #####
rho = 6.371E6
phi_0 = 48.764246

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

##### Fonctions de distance #####
def yj_yi(phij,phii):
    return rho*2*np.pi*(phij-phii)/360

def xj_xi(lambdaj,lambdai):
    return rho*math.cos(2*np.pi*phi_0/360)*2*np.pi*(lambdaj-lambdai)/360

def distM(phii, phij, lambdai, lambdaj):
    deltax = xj_xi(lambdaj,lambdai)
    deltay = yj_yi(phij,phii)
    return abs(deltax)+abs(deltay)

def distE(deltax, deltay, lambdai, lambdaj, phii, phij):
    deltax = xj_xi(lambdaj,lambdai)
    deltay = yj_yi(phij,phii)
    return math.sqrt(deltax**2+deltay**2)


##### Fonctions de coût #####
#R une liste qui contient les routes dans le bon ordre

#fonction de cout non linéarisée
def c(R):
    cost = 0
    for i in range(len(R)):
        family = R[i][0]
        c_rental = vehicles[family-1]["rental_cost"]

        c_fuel = distM(0, R[1])
        for k in range(1, len(R[i])-1):
            c_fuel += distM(R[k], R[k+1])
            if k == len(R[i]-2):
                c_fuel += distM(R[k+1], 0)
        c_fuel = vehicles[family-1]["fuel_cost"] * c_fuel

        dist_euclid = []
        for k in  range(1, len(R[i])):
            for l in range(1, len(R[i])):
                if k != l :
                    dist_euclid.append(distE(k,l))
        c_radius = vehicles[family-1]["radius_cost"]*max(dist_euclid)/2 

        return c_rental + c_fuel + c_radius
    




