import pandas as pd
import numpy as np
import math
from itertools import combinations

# -----------------
# Data Loading
# -----------------

def load_data(instance_name):
    """Loads all data for a given instance using pandas."""
    instance_path = f"sujet/instances/{instance_name}.csv"
    vehicles_path = "sujet/instances/vehicles.csv"

    orders_df = pd.read_csv(instance_path)
    vehicles_df = pd.read_csv(vehicles_path)

    depot = orders_df[orders_df['id'] == 0].iloc[0]
    orders = orders_df[orders_df['id'] != 0].copy()

    # Convert lat/lon to x/y coordinates
    orders['x'], orders['y'] = geo_to_xy(orders['latitude'], orders['longitude'], depot['latitude'], depot['longitude'])
    depot_coords = {'x': 0, 'y': 0, 'latitude': depot['latitude'], 'longitude': depot['longitude']}

    return vehicles_df, orders, depot_coords

# -----------------
# Helper Functions
# -----------------

def geo_to_xy(lat, lon, ref_lat, ref_lon):
    """Converts latitude and longitude to x and y coordinates in meters."""
    R = 6371000  # Earth radius in meters
    x = R * (lon - ref_lon) * np.pi / 180 * np.cos(ref_lat * np.pi / 180)
    y = R * (lat - ref_lat) * np.pi / 180
    return x, y

def euclidean_distance(p1, p2):
    """Calculates Euclidean distance between two points (can be orders or depot)."""
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def manhattan_distance(p1, p2):
    """Calculates Manhattan distance between two points (can be orders or depot)."""
    return np.abs(p1['x'] - p2['x']) + np.abs(p1['y'] - p2['y'])

def travel_time(vehicle, distance, departure_time):
    """Calculates travel time between two locations for a given vehicle."""
    T = 86400
    w = 2 * np.pi / T
    time_factor = sum(
        vehicle[f'fourier_cos_{n}'] * np.cos(n * w * departure_time) +
        vehicle[f'fourier_sin_{n}'] * np.sin(n * w * departure_time)
        for n in range(4)
    )
    return (distance / vehicle['speed']) * time_factor

def recalculate_route_state_from_ids(order_ids, vehicle, orders_by_id, depot):
    """
    Recalculates the time and weight of a route and checks its validity.
    Returns (is_valid, final_time, final_weight).
    """
    if not order_ids:
        return True, 0, 0
        
    current_weight = 0
    current_time = 0
    last_location = depot

    for order_id in order_ids:
        order = orders_by_id.loc[order_id]
        
        # Check capacity
        current_weight += order['order_weight']
        if current_weight > vehicle['max_capacity']:
            return False, -1, -1

        # Check time window
        distance = manhattan_distance(last_location, order)
        time_to_travel = travel_time(vehicle, distance, current_time)
        
        arrival_time = current_time + time_to_travel
        if not (last_location['x'] == depot['x'] and last_location['y'] == depot['y']):
             arrival_time += vehicle['parking_time']

        if arrival_time > order['window_end']:
            return False, -1, -1
        
        current_time = max(arrival_time, order['window_start']) + order['delivery_duration']
        last_location = order

    return True, current_time, current_weight

# -----------------
# Cost Calculation
# -----------------

def calculate_route_cost(route, vehicle, orders_by_id, depot):
    """Calculates the cost of a single route."""
    cost = 0
    order_ids = route[1:]

    # Rental cost
    cost += vehicle['rental_cost']

    # Fuel cost
    route_distance = 0
    last_location = depot
    for order_id in order_ids:
        order = orders_by_id.loc[order_id]
        route_distance += manhattan_distance(last_location, order)
        last_location = order
    route_distance += manhattan_distance(last_location, depot)
    cost += vehicle['fuel_cost'] * route_distance

    # Radius cost
    if len(order_ids) > 1:
        route_orders = orders_by_id.loc[order_ids]
        max_dist = 0
        # Use itertools.combinations to find the max distance between any two points
        for p1, p2 in combinations(route_orders.to_dict('records'), 2):
            dist = euclidean_distance(p1, p2)
            if dist > max_dist:
                max_dist = dist
        cost += vehicle['radius_cost'] * (0.5 * max_dist)
    
    return cost

def calculate_total_cost(solution, vehicles_df, orders, depot):
    """Calculates the total cost of a solution."""
    total_cost = 0
    orders_by_id = orders.set_index('id')
    vehicles_by_family = vehicles_df.set_index('family')

    for route in solution:
        family_id = route[0]
        vehicle = vehicles_by_family.loc[family_id]
        total_cost += calculate_route_cost(route, vehicle, orders_by_id, depot)

    return total_cost

# -----------------
# Main Algorithm
# -----------------

def solve_instance(instance_name):
    """Solves a single instance of the VRP using a cheapest insertion heuristic."""
    vehicles_df, orders, depot = load_data(instance_name)
    orders_by_id = orders.set_index('id')
    
    solution_routes = []
    unassigned_orders_mask = np.ones(len(orders), dtype=bool)

    # Sort vehicles by a heuristic (e.g., cheaper rental cost first)
    vehicles_df = vehicles_df.sort_values(by=['rental_cost', 'max_capacity']).reset_index(drop=True)

    # --- CONSTRUCTION PHASE (Cheapest Insertion) ---
    for _, vehicle in vehicles_df.iterrows():
        # Try to build routes with this vehicle type as long as there are unassigned orders
        while np.any(unassigned_orders_mask):
            current_route_ids = []
            
            # Iteratively add the 'cheapest' order to the current route
            while True:
                best_order_idx = -1
                best_insert_pos = -1
                min_insertion_cost = float('inf')

                _, _, old_weight = recalculate_route_state_from_ids(current_route_ids, vehicle, orders_by_id, depot)
                
                cost_before = 0
                if current_route_ids:
                    cost_before = calculate_route_cost([vehicle['family']] + current_route_ids, vehicle, orders_by_id, depot)

                # Find the best order to insert
                for order_idx in np.where(unassigned_orders_mask)[0]:
                    order_to_insert = orders.iloc[order_idx]

                    # Quick check on weight before trying all positions
                    if old_weight + order_to_insert['order_weight'] > vehicle['max_capacity']:
                        continue

                    # Find the best insertion position for this order
                    for i in range(len(current_route_ids) + 1):
                        temp_route_ids = current_route_ids[:i] + [order_to_insert['id']] + current_route_ids[i:]
                        is_valid, _, _ = recalculate_route_state_from_ids(temp_route_ids, vehicle, orders_by_id, depot)

                        if is_valid:
                            # Calculate cost increase
                            cost_after = calculate_route_cost([vehicle['family']] + temp_route_ids, vehicle, orders_by_id, depot)
                            
                            insertion_cost = cost_after
                            if current_route_ids:
                                # If route is not empty, the cost increase is the difference
                                insertion_cost -= cost_before
                            
                            if insertion_cost < min_insertion_cost:
                                min_insertion_cost = insertion_cost
                                best_order_idx = order_idx
                                best_insert_pos = i

                # If a valid insertion was found, add the order to the route
                if best_order_idx != -1:
                    order_id_to_insert = orders.iloc[best_order_idx]['id']
                    current_route_ids.insert(best_insert_pos, order_id_to_insert)
                    unassigned_orders_mask[best_order_idx] = False
                else:
                    # No more orders can be added to this route
                    break
            
            # If a route was built, add it to the solution
            if current_route_ids:
                # Ensure all values are integers for the solution file
                solution_routes.append([int(vehicle['family'])] + [int(oid) for oid in current_route_ids])
            else:
                # If no route could be started with this vehicle, move to the next vehicle type
                break
        
        # If all orders are assigned, we can stop creating new routes
        if not np.any(unassigned_orders_mask):
            break
            
    return solution_routes

# -----------------
# Output
# -----------------

def write_solution(solution, filename):
    """Writes the solution to a CSV file."""
    if not solution:
        with open(filename, 'w') as f:
            f.write('family,order_1\n')
        return

    max_len = max(len(row) for row in solution) if solution else 0
    
    header = ['family'] + [f'order_{i}' for i in range(1, max_len)]

    solution_dict_list = []
    for row in solution:
        row_dict = {}
        padded_row = row + [None] * (max_len - len(row))
        for i, val in enumerate(padded_row):
            if i < len(header):
                # Ensure values are integers where possible
                row_dict[header[i]] = int(val) if val is not None else None
        solution_dict_list.append(row_dict)

    df = pd.DataFrame(solution_dict_list, columns=header)
    
    # Use a nullable integer type that can handle NaNs
    for col in df.columns:
        df[col] = df[col].astype('Int64')

    df.to_csv(filename, index=False, na_rep='')

# -----------------
# Main
# -----------------

if __name__ == "__main__":
    total_cost_all_instances = 0
    for i in range(1, 11):
        instance_name = f"instance_{i:02d}"
        print(f"Solving {instance_name}...")
        
        # We need to load data here to pass to the cost function later
        vehicles_df, orders, depot = load_data(instance_name)
        
        solution = solve_instance(instance_name)
        
        output_filename = f"solution_{instance_name}.csv"
        write_solution(solution, output_filename)
        print(f"Solution for {instance_name} written to {output_filename}")
        
        # Calculate and print the cost for the instance
        cost = calculate_total_cost(solution, vehicles_df, orders, depot)
        total_cost_all_instances += cost
        print(f"Cost for {instance_name}: {cost}")
        print("-" * 20)

    print(f"\nTotal cost for all instances: {total_cost_all_instances}")
    print("All instances solved.")