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

instances = []
folder="sujet/instances/"

# Boucle sur les 9 fichiers
for i in range(1, 11):
    file_path = f"{folder}instance_{i:02d}.csv"  # format : instance_01.csv, instance_02.csv, etc.

    # Charger le fichier
    data_instancei = pd.read_csv(file_path)

    # Convertir en liste de dictionnaires (une entrée par point)
    instancei = [row[1:].to_dict() for _, row in data_instancei.iterrows()]

    # Ajouter à la liste principale
    instances.append(instancei)

def yj_yi(phij,phii):
    return rho*2*np.pi*(phij-phii)/360
def xj_xi(lambdaj,lambdai):
    return rho*math.cos(2*np.pi*phi_0/360)*2*np.pi*(lambdaj-lambdai)/360

def distE(i,j,A):
    deltax = xj_xi(instances[A][j]["longitude"],instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"],instances[A][i]["latitude"])
    return math.sqrt(deltax**2+deltay**2)

def distM(i,j,A):
    deltax = xj_xi(instances[A][j]["longitude"],instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"],instances[A][i]["latitude"])
    return abs(deltax)+abs(deltay)

def traveltimes(i,j,f,t,A):
    gamma=0
    for i in range (4):
        gamma+=vehicles[f][f"fourier_cos_{n}"]*math.cos(n*2*math.pi*t/86400)+vehicules[f][f"fourier_sin_{n}"]*math.sin(n*2*math.pi*t/86400)
    return (distM(i,j,A)/vehicles[f]["speed"]+vehicles[f]["parking_time"])*gamma
