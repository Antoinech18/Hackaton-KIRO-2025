import pandas as pd
import numpy as np
import math
import pulp 
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

vehicles = []
for idx, row in data_vehicles.iterrows():
    vehicle_dict = {c: row[c] for c in columns}
    vehicles.append(vehicle_dict)

instances = []
folder = "sujet/instances/"

# Chargement de chaque instance
for i in range(1, 11):
    file_path = f"{folder}instance_{i:02d}.csv"
    data_instancei = pd.read_csv(file_path)
    instancei = data_instancei.to_dict(orient="records")  # garde toutes les colonnes
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

#définition du model
A = 0                      # on travaille sur instance_01.csv
instance = instances[A]    # liste de dicts
# --- Données du modèle ---
clients = [i for i in range(len(instance))]  # 0 = dépôt, le reste = commandes
familles = [1, 2, 3]        # familles de véhicules

w = {i: instance[i]["order_weight"] if not pd.isna(instance[i]["order_weight"]) else 0 for i in clients}
l = {i: instance[i]["delivery_duration"] if not pd.isna(instance[i]["delivery_duration"]) else 0 for i in clients}
tmin = {i: instance[i]["window_start"] if not pd.isna(instance[i]["window_start"]) else 0 for i in clients}
tmax = {i: instance[i]["window_end"] if not pd.isna(instance[i]["window_end"]) else 86400 for i in clients}

# On suppose que tous les trajets possibles sont autorisés sauf i == j
arcs = [(i, j, f) for i in clients for j in clients if i != j for f in familles]

# --- Création du modèle ---
model = pulp.LpProblem("Califrais_Routing", pulp.LpMinimize)
# --- Variables de décision ---
x = pulp.LpVariable.dicts("x", arcs, lowBound=0, upBound=1, cat="Binary")

t = pulp.LpVariable.dicts("arrival_time", clients, lowBound=0)

# Charge dans le véhicule lorsque l'on quitte i
Umax = max(v["max_capacity"] for v in vehicles)
load = pulp.LpVariable.dicts("load", clients, lowBound=0, upBound=Umax)

# Temps et charge au dépôt
model += t[0] == 0, "time_at_depot"
model += load[0] == 0, "load_at_depot"
# --- Contrainte 1 : chaque client est visité exactement une fois ---
for j in clients:
    if j != 0:  # on exclut le dépôt
        model += (
            pulp.lpSum(x[i, j, f] for i in clients if i != j for f in familles) == 1,
            f"visit_once_client_{j}"
        )


for j in clients:
    if j != 0:
        for f in familles:
            model += (
                pulp.lpSum(x[i, j, f] for i in clients if i != j)
                ==
                pulp.lpSum(x[j, k, f] for k in clients if k != j),
                f"flow_balance_{j}_{f}"
            )

for f in familles:
    model += (
        pulp.lpSum(x[0, j, f] for j in clients if j != 0)
        ==
        pulp.lpSum(x[i, 0, f] for i in clients if i != 0),
        f"depot_flow_{f}"
    )

for i in clients:
    for j in clients:
        if i != j:
            for f in familles:
                Qf = vehicles[f-1]["max_capacity"]  # f-1 car vehicles est 0-based
                model += (
                    load[j] >= load[i] + w[j] - Qf * (1 - x[i, j, f]),
                    f"capacity_arc_{i}_{j}_{f}"
                )

for i in clients:
    if i != 0:
        model += t[i] >= tmin[i], f"time_window_min_{i}"
        model += t[i] <= tmax[i], f"time_window_max_{i}"

M = 10**6

for i in clients:
    for j in clients:
        if i != j:
            for f in familles:
                speed_f = vehicles[f-1]["speed"]
                park_f = vehicles[f-1]["parking_time"]
                tau_ij = distM(i, j, A) / speed_f + park_f
                li = l[i] if not pd.isna(l[i]) else 0.0

                model += (
                    t[j] >= t[i] + li + tau_ij - M * (1 - x[i, j, f]),
                    f"time_seq_{i}_{j}_{f}"
                )

model += pulp.lpSum(
    vehicles[f-1]["fuel_cost"] * distM(i, j, A) * x[i, j, f]
    for i in clients for j in clients if i != j for f in familles
)
