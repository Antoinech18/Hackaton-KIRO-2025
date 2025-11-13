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

##### Fonctions géométriques #####

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

# Version correcte de traveltimes (NON utilisée dans le modèle, car non linéaire)
def traveltimes(i,j,f,t,A):
    """
    Retourne tau_f(i,j|t) exact (non linéaire) : juste pour référence / debug.
    f est ici la famille 1,2,3 -> on va chercher vehicles[f-1]
    """
    v = vehicles[f-1]
    base = distM(i,j,A) / v["speed"] + v["parking_time"]
    gamma = 0.0
    for n in range(4):
        gamma += v[f"fourier_cos_{n}"] * math.cos(n*2*math.pi*t/86400) \
               + v[f"fourier_sin_{n}"] * math.sin(n*2*math.pi*t/86400)
    return base * gamma

##### Choix de l'instance #####
A = 0                      # on travaille sur instance_01.csv
instance = instances[A]    # liste de dicts

##### Données du modèle #####
clients = [i for i in range(len(instance))]  # 0 = dépôt, le reste = commandes
familles = [1, 2, 3]        # familles de véhicules

w = {i: (instance[i]["order_weight"] if not pd.isna(instance[i]["order_weight"]) else 0.0)
     for i in clients}
l = {i: (instance[i]["delivery_duration"] if not pd.isna(instance[i]["delivery_duration"]) else 0.0)
     for i in clients}
tmin = {i: (instance[i]["window_start"] if not pd.isna(instance[i]["window_start"]) else 0.0)
        for i in clients}
tmax = {i: (instance[i]["window_end"] if not pd.isna(instance[i]["window_end"]) else 86400.0)
        for i in clients}

# Pré-calcul de gamma_max pour chaque famille (borne supérieure sur γ_f(t))
gamma_max = []
for v in vehicles:
    g = 0.0
    for n in range(4):
        g += abs(v[f"fourier_cos_{n}"]) + abs(v[f"fourier_sin_{n}"])
    gamma_max.append(g)

# On suppose que tous les trajets possibles sont autorisés sauf i == j
arcs = [(i, j, f) for i in clients for j in clients if i != j for f in familles]

##### Création du modèle #####
model = pulp.LpProblem("Califrais_Routing", pulp.LpMinimize)
radius_term = pulp.lpSum(
    vehicles[f-1]["radius_cost"] * distE(0, j, A) * x[0, j, f]
    for j in clients if j != 0 for f in familles
)

# Variables de décision
x = pulp.LpVariable.dicts("x", arcs, lowBound=0, upBound=1, cat="Binary")

# Temps d'arrivée
t = pulp.LpVariable.dicts("arrival_time", clients, lowBound=0)

# Charge dans le véhicule par sommet et par famille
load_keys = [(i,f) for i in clients for f in familles]
load = pulp.LpVariable.dicts("load", load_keys, lowBound=0)

# Temps au dépôt (0)
model += t[0] == 0, "time_at_depot"

# Charge au dépôt nulle pour chaque famille
for f in familles:
    model += load[(0,f)] == 0, f"load_at_depot_f{f}"

# Bornes sup sur load[i,f] : pas plus que la capacité de la famille
for i in clients:
    for f in familles:
        Qf = vehicles[f-1]["max_capacity"]
        model += load[(i,f)] <= Qf, f"load_cap_{i}_{f}"

##### Contraintes #####

# 1) Chaque client est visité exactement une fois (hors dépôt)
for j in clients:
    if j != 0:  # on exclut le dépôt
        model += (
            pulp.lpSum(x[i, j, f] for i in clients if i != j for f in familles) == 1,
            f"visit_once_client_{j}"
        )

# 2) Conservation des flux pour chaque client et chaque famille
for j in clients:
    if j != 0:
        for f in familles:
            model += (
                pulp.lpSum(x[i, j, f] for i in clients if i != j)
                ==
                pulp.lpSum(x[j, k, f] for k in clients if k != j),
                f"flow_balance_{j}_{f}"
            )

# 3) Dépôt : départs = retours pour chaque famille
for f in familles:
    model += (
        pulp.lpSum(x[0, j, f] for j in clients if j != 0)
        ==
        pulp.lpSum(x[i, 0, f] for i in clients if i != 0),
        f"depot_flow_{f}"
    )

# 4) Capacité via variables de charge (MTZ par famille)
for i in clients:
    for j in clients:
        if i != j:
            for f in familles:
                Qf = vehicles[f-1]["max_capacity"]
                model += (
                    load[(j,f)] >= load[(i,f)] + w[j] - Qf * (1 - x[i, j, f]),
                    f"capacity_arc_{i}_{j}_{f}"
                )

# 5) Fenêtres de temps
for i in clients:
    if i != 0:
        model += t[i] >= tmin[i], f"time_window_min_{i}"
        model += t[i] <= tmax[i], f"time_window_max_{i}"

# 6) Séquencement temporel avec Big-M et borne linéaire sur les temps
M = 10**7

for i in clients:
    for j in clients:
        if i != j:
            for f in familles:
                v = vehicles[f-1]
                base_tau = distM(i, j, A) / v["speed"] + v["parking_time"]
                tau_ij = base_tau * gamma_max[f-1]  # borne supérieure sur tau_f(i,j|t)
                li = l[i]

                model += (
                    t[j] >= t[i] + li + tau_ij - M * (1 - x[i, j, f]),
                    f"time_seq_{i}_{j}_{f}"
                )

##### Fonction objectif (linéaire) #####
# Ici : coût carburant + coût de location (simple)
# - fuel_cost * distance
# - rental_cost * nb de véhicules (approx = nb de départs depuis le dépôt)

fuel_term = pulp.lpSum(
    vehicles[f-1]["fuel_cost"] * distM(i, j, A) * x[i, j, f]
    for i in clients for j in clients if i != j for f in familles
)

rental_term = pulp.lpSum(
    vehicles[f-1]["rental_cost"] * pulp.lpSum(x[0, j, f] for j in clients if j != 0)
    for f in familles
)

# On laisse de côté le radius_cost exact (non linéaire) ou on l'approximera plus tard.
model += fuel_term + rental_term + radius_term 


##### Résolution #####
status = model.solve(pulp.PULP_CBC_CMD(msg=1))
print("Status:", pulp.LpStatus[status])

##### Construction du fichier routes.csv #####

# Si pas de solution, on s'arrête proprement
if pulp.LpStatus[status] not in ["Optimal", "Feasible"]:
    print("Pas de solution exploitable, pas de routes.csv généré.")
else:
    # 1) Extraire les arcs utilisés
    EPS = 1e-5

    successors = {(i, f): [] for i in clients for f in familles}
    for (i, j, f) in arcs:
        val = pulp.value(x[i, j, f])
        if val is not None and val > EPS:
            successors[(i, f)].append(j)

    # 2) Reconstruire les routes à partir des arcs sortant du dépôt
    routes = []  # liste de dicts {"family": f, "nodes": [i1,i2,...,in]}

    for f in familles:
        starts = list(successors[(0, f)])  # clients atteints depuis le dépôt
        for start in starts:
            route_nodes = []
            current = start
            visited = set()

            # Suivre la chaîne jusqu'au retour au dépôt ou blocage
            while current != 0 and current not in visited:
                visited.add(current)
                if current != 0:
                    route_nodes.append(current)
                succs = successors.get((current, f), [])
                if len(succs) == 0:
                    break
                current = succs[0]  # on suppose au plus un successeur actif

            if len(route_nodes) > 0:
                routes.append({
                    "family": f,
                    "nodes": route_nodes
                })

    # 3) Conversion des indices en IDs de commandes
    for r in routes:
        r["order_ids"] = [instance[i]["id"] for i in r["nodes"]]

    # 4) Construction du DataFrame routes.csv
    if len(routes) > 0:
        max_len = max(len(r["order_ids"]) for r in routes)
    else:
        max_len = 0

    cols = ["family"] + [f"order_{k+1}" for k in range(max_len)]
    rows = []

    for r in routes:
        row = [r["family"]] + r["order_ids"]
        while len(row) < len(cols):
            row.append("")
        rows.append(row)

    routes_df = pd.DataFrame(rows, columns=cols)
    routes_df.to_csv("routes.csv", index=False)
    print("Fichier routes.csv généré.")
