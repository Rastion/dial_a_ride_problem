import os
import sys
import json
import math
import random
from qubots.base_problem import BaseProblem

# Helper functions to read the instance from a JSON file
def read_data(filename):
    with open(filename) as f:
        return json.load(f)

def read_input_darp(instance_file):
    instance = read_data(instance_file)

    nb_clients = instance["nbClients"]
    nb_nodes = instance["nbNodes"]
    nb_vehicles = instance["nbVehicles"]
    depot_tw_end = instance["depot"]["twEnd"]
    capacity = instance["capacity"]
    scale = instance["scale"]

    # There are 2*nbClients service nodes: first nbClients are pickups, next nbClients are deliveries.
    quantities = [-1 for _ in range(2 * nb_clients)]
    distances = instance["distanceMatrix"]
    starts = [-1.0 for _ in range(2 * nb_clients)]
    ends = [-1.0 for _ in range(2 * nb_clients)]
    loading_times = [-1.0 for _ in range(2 * nb_clients)]
    max_travel_times = [-1.0 for _ in range(2 * nb_clients)]
    for k in range(nb_clients):
        quantities[k] = instance["clients"][k]["nbClients"]
        quantities[k + nb_clients] = -instance["clients"][k]["nbClients"]

        starts[k] = instance["clients"][k]["pickup"]["start"]
        ends[k] = instance["clients"][k]["pickup"]["end"]

        starts[k + nb_clients] = instance["clients"][k]["delivery"]["start"]
        ends[k + nb_clients] = instance["clients"][k]["delivery"]["end"]

        loading_times[k] = instance["clients"][k]["pickup"]["loadingTime"]
        loading_times[k + nb_clients] = instance["clients"][k]["delivery"]["loadingTime"]

        max_travel_times[k] = instance["clients"][k]["pickup"]["maxTravelTime"]
        max_travel_times[k + nb_clients] = instance["clients"][k]["delivery"]["maxTravelTime"]

    factor = 1.0 / (scale * instance["speed"])

    nb_total_nodes = nb_nodes  # total nodes including depot? (The JSON uses an external depot)
    distance_warehouse = [-1.0 for _ in range(nb_total_nodes)]
    time_warehouse = [-1.0 for _ in range(nb_total_nodes)]
    distance_matrix = [[-1.0 for _ in range(nb_total_nodes)] for _ in range(nb_total_nodes)]
    time_matrix = [[-1.0 for _ in range(nb_total_nodes)] for _ in range(nb_total_nodes)]
    for i in range(nb_total_nodes):
        distance_warehouse[i] = distances[0][i+1]
        time_warehouse[i] = distance_warehouse[i] * factor
        for j in range(nb_total_nodes):
            distance_matrix[i][j] = distances[i+1][j+1]
            time_matrix[i][j] = distance_matrix[i][j] * factor

    return (nb_clients, nb_nodes, nb_vehicles, depot_tw_end, capacity, scale,
            quantities, starts, ends, loading_times, max_travel_times,
            distance_warehouse, time_warehouse, distance_matrix, time_matrix)

# A large penalty value for infeasible solutions
PENALTY = 1e9

class DARPProblem(BaseProblem):
    """
    Dial-A-Ride Problem (DARP):
    
    A fleet of vehicles must pick up and deliver clients. Each client is defined by a pickup
    time window and a delivery time window, as well as loading times and a maximum travel time.
    Vehicles start and end their routes at a depot and have a limited capacity.
    
    Instance Data:
      - nb_clients: number of clients (each with one pickup and one delivery, for a total of 2*nb_clients service nodes)
      - nb_nodes: total number of service nodes (should equal 2*nb_clients)
      - nb_vehicles: number of available vehicles
      - depot_tw_end: maximum allowed return time at the depot
      - capacity: vehicle capacity (maximum cumulative quantity)
      - scale: a scaling factor for converting distances into time via the vehicle speed
      - quantities: list of 2*nb_clients numbers (pickup demands are positive; deliveries are negative)
      - starts, ends: lists of earliest and latest allowed service times for each node
      - loading_times: service (loading) time at each node
      - max_travel_times: maximum allowed travel time between pickup and delivery for each client
      - distance_warehouse: distances from the depot to each service node
      - time_warehouse: travel times from the depot to each service node
      - distance_matrix: 2D matrix of distances between service nodes
      - time_matrix: 2D matrix of travel times between service nodes
    
    Candidate Solution:
      A dictionary with a key "routes" mapping to a list of routes, one per vehicle.
      Each route is a list of service node indices (integers in 0..(2*nb_clients–1)).
      (The candidate is expected to respect that for each client, both its pickup (index k)
       and its delivery (index k+nb_clients) appear in the same route and the pickup occurs before
       the delivery.)
    
    Objective:
      For each vehicle route, the departure times are computed as:
         - For the first node: t0 = max( starts[node0], depot_start + time_warehouse[node0] ) + loading_times[node0]
         - For subsequent nodes: t[i] = max( starts[node], t[i-1] + time_matrix[prev][node] ) + loading_times[node]
         (Here we assume depot_start = 0 and no waiting time.)
      Then:
         - Route lateness is the sum over nodes of max(0, t[i] – loading_times[node] – ends[node])
         - Home lateness is max(0, t[last] + time_warehouse[node_last] – depot_tw_end)
         - For each client, client lateness = max( (delivery_time – loading_time(delivery)) – pickup_time – max_travel_time, 0 )
         - Route distance = distance from depot to first node + distances along the route + distance from last node back to depot.
      The overall objective is lexicographically minimizing:
         (total lateness (including home lateness), total client lateness, total distance/scale).
      In this implementation we combine these using scaling factors so that violations incur a large cost.
    """
    def __init__(self, instance_file=None, **kwargs):
        if instance_file is not None:
            self._load_instance(instance_file)
        else:
            required = ["nb_clients", "nb_nodes", "nb_vehicles", "depot_tw_end", "capacity", "scale",
                        "quantities", "starts", "ends", "loading_times", "max_travel_times",
                        "distance_warehouse", "time_warehouse", "distance_matrix", "time_matrix"]
            for r in required:
                if r not in kwargs:
                    raise ValueError("Missing parameter: " + r)
            self.nb_clients = kwargs["nb_clients"]
            self.nb_nodes = kwargs["nb_nodes"]
            self.nb_vehicles = kwargs["nb_vehicles"]
            self.depot_tw_end = kwargs["depot_tw_end"]
            self.capacity = kwargs["capacity"]
            self.scale = kwargs["scale"]
            self.quantities = kwargs["quantities"]
            self.starts = kwargs["starts"]
            self.ends = kwargs["ends"]
            self.loading_times = kwargs["loading_times"]
            self.max_travel_times = kwargs["max_travel_times"]
            self.distance_warehouse = kwargs["distance_warehouse"]
            self.time_warehouse = kwargs["time_warehouse"]
            self.distance_matrix = kwargs["distance_matrix"]
            self.time_matrix = kwargs["time_matrix"]

    def _load_instance(self, filename):
        (self.nb_clients, self.nb_nodes, self.nb_vehicles, self.depot_tw_end,
         self.capacity, self.scale, self.quantities, self.starts, self.ends,
         self.loading_times, self.max_travel_times, self.distance_warehouse,
         self.time_warehouse, self.distance_matrix, self.time_matrix) = read_input_darp(filename)

    def evaluate_solution(self, solution) -> float:
        # The candidate solution must be a dict with key "routes"
        if not isinstance(solution, dict) or "routes" not in solution:
            return PENALTY
        routes = solution["routes"]
        if not isinstance(routes, list) or len(routes) != self.nb_vehicles:
            return PENALTY

        # Check that every service node (0 .. 2*nb_clients-1) appears exactly once
        total_nodes = 2 * self.nb_clients
        visited = [False] * total_nodes
        for route in routes:
            if not isinstance(route, list):
                return PENALTY
            for node in route:
                if not isinstance(node, int) or node < 0 or node >= total_nodes:
                    return PENALTY
                if visited[node]:
                    return PENALTY
                visited[node] = True
        if not all(visited):
            return PENALTY

        total_lateness = 0.0
        total_route_distance = 0.0
        # For each vehicle, compute the departure times along its route.
        vehicle_times = []  # list of lists of times per route
        for route in routes:
            if not route:
                vehicle_times.append([])
                continue
            # Assume depot start = 0 and waiting time = 0.
            t0 = max(self.starts[route[0]], 0 + self.time_warehouse[route[0]]) + self.loading_times[route[0]]
            times = [t0]
            # Compute route times sequentially
            for i in range(1, len(route)):
                travel = self.time_matrix[route[i-1]][route[i]]
                t = max(self.starts[route[i]], times[-1] + travel) + self.loading_times[route[i]]
                times.append(t)
            vehicle_times.append(times)
            # Route lateness: for each node, lateness = max(0, t - loading_time - ends)
            route_lateness = sum(max(0, t - self.loading_times[route[i]] - self.ends[route[i]])
                                 for i, t in enumerate(times))
            # Home lateness: extra time after finishing route plus travel back to depot
            home_lateness = max(0, times[-1] + self.time_warehouse[route[-1]] - self.depot_tw_end)
            total_lateness += (route_lateness + home_lateness)
            # Route distance: depot->first + sum of distances + last->depot
            rd = self.distance_warehouse[route[0]] + \
                 sum(self.distance_matrix[route[i-1]][route[i]] for i in range(1, len(route))) + \
                 self.distance_warehouse[route[-1]]
            total_route_distance += rd

            # Capacity check: the cumulative sum of quantities must be between 0 and capacity.
            cum = 0
            for node in route:
                cum += self.quantities[node]
                if cum < 0 or cum > self.capacity:
                    return PENALTY

        # Now check client-specific constraints:
        # For each client (0 to nb_clients-1), find its pickup (node k) and delivery (node k+nb_clients)
        total_client_lateness = 0.0
        for client in range(self.nb_clients):
            pickup_found = delivery_found = False
            pickup_time = delivery_time = None
            for r_idx, route in enumerate(routes):
                if client in route and (client + self.nb_clients) in route:
                    pickup_index = route.index(client)
                    delivery_index = route.index(client + self.nb_clients)
                    if pickup_index >= delivery_index:
                        return PENALTY
                    pickup_time = vehicle_times[r_idx][pickup_index]
                    # As in the model, subtract the loading time at delivery
                    delivery_time = vehicle_times[r_idx][delivery_index] - self.loading_times[client + self.nb_clients]
                    pickup_found = delivery_found = True
                    break
            if not (pickup_found and delivery_found):
                return PENALTY
            travel_time = delivery_time - pickup_time
            client_lateness = max(travel_time - self.max_travel_times[client], 0)
            total_client_lateness += client_lateness

        # Let trucks_used be the count of nonempty routes.
        trucks_used = sum(1 for route in routes if route)
        # To combine the three objectives lexicographically, we use large multipliers.
        # Here we choose:
        L1 = 1e6  # penalty per truck used
        L2 = 1e3  # weight for total client lateness
        L3 = 1e-3 # weight for total distance (after scaling)
        overall = trucks_used * L1 + total_lateness + total_client_lateness * L2 + (total_route_distance / self.scale) * L3
        return overall

    def random_solution(self):
        """
        Generates a random feasible solution:
          - For each client, assign its pickup and delivery to a random vehicle.
          - In each vehicle’s route, order the assigned clients by a random permutation and then insert
            each client’s pickup immediately before its delivery.
        """
        total_nodes = 2 * self.nb_clients
        # Create empty routes for each vehicle
        routes = [[] for _ in range(self.nb_vehicles)]
        # For each client, randomly choose a vehicle and append a tuple (client, random_key)
        assignments = [[] for _ in range(self.nb_vehicles)]
        for client in range(self.nb_clients):
            vehicle = random.randrange(self.nb_vehicles)
            # Generate a random key to sort clients in this vehicle
            assignments[vehicle].append((client, random.random()))
        # For each vehicle, sort by key and then create a route by inserting pickup and delivery (pickup before delivery)
        for v in range(self.nb_vehicles):
            # sort clients assigned to vehicle v
            sorted_clients = [c for c, _ in sorted(assignments[v], key=lambda x: x[1])]
            # In the route, insert pickup then delivery for each client
            route = []
            for c in sorted_clients:
                route.append(c)            # pickup node (c)
                route.append(c + self.nb_clients)  # delivery node (c+nb_clients)
            routes[v] = route
        return {"routes": routes}
