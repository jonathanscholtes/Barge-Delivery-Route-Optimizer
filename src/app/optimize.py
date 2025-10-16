# optimize.py
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from datetime import datetime


class Optimizer:
    """
    Optimizer for weekly dispatch planning using CVRPTW (Capacitated Vehicle Routing Problem with Time Windows).

    Attributes:
        forecast_df (pd.DataFrame): Forecasted weekly demand with columns ['site_id', 'week_start', 'forecast_units'].
        site_specs_file (str): Path to CSV with site specifications (open/close times, service times, etc.).
        travel_file (str): Path to CSV with travel times between sites.
        barge_file (str): Path to CSV with barge specifications (capacity, working hours, loading rate).

    Methods:
        run(week_start): Generates a dispatch route for the specified week_start date.
    """

    def __init__(self, forecast_df, site_specs_file, travel_file, barge_file):
        self.forecast_df = forecast_df
        self.site_specs_file = site_specs_file
        self.travel_file = travel_file
        self.barge_file = barge_file

    def __build_week_input(self, week_start_date):
        """
        Prepare site nodes and travel times for the specified week.

        Returns:
            df_nodes (pd.DataFrame): Sites with forecasted demand and service info.
            tt_df (pd.DataFrame): Travel times between sites.
        """
        target_date = pd.to_datetime(week_start_date)
        self.forecast_df['week_start'] = pd.to_datetime(self.forecast_df['week_start'], errors='coerce')

        # Filter forecast for this week
        week = self.forecast_df[self.forecast_df['week_start'].dt.date == target_date.date()]
        if week.empty:
            print(f"No forecast rows for week_start {target_date.date()}")
            return pd.DataFrame(), pd.DataFrame()

        # Aggregate demand per site
        demand = week.groupby('site_id')['forecast_units'].sum().reset_index()

        # Load site specs and ensure depot exists
        sites = pd.read_csv(self.site_specs_file)
        if 'PORT0' not in sites['site_id'].values:
            depot_row = {'site_id': 'PORT0', 'lat': None, 'lon': None,
                         'open_time': '00:00', 'close_time': '23:59',
                         'service_time_minutes': 0, 'max_visit_volume_units': 0}
            sites = pd.concat([pd.DataFrame([depot_row]), sites], ignore_index=True)

        df_nodes = pd.merge(sites, demand, on='site_id', how='right').fillna(0)

        # Load travel times
        tt_df = pd.read_csv(self.travel_file)

        # Check for missing travel edges
        missing_edges = []
        for site in df_nodes['site_id']:
            if site == 'PORT0':
                continue
            if not ((tt_df['from_site'] == 'PORT0') & (tt_df['to_site'] == site)).any():
                missing_edges.append(f"PORT0->{site}")
            if not ((tt_df['from_site'] == site) & (tt_df['to_site'] == 'PORT0')).any():
                missing_edges.append(f"{site}->PORT0")
        if missing_edges:
            print("Warning: missing travel-time edges:", missing_edges)

        return df_nodes, tt_df



    def __create_data_model(self, df_nodes, tt_df, barge_capacity_units, depot_id='PORT0'):
        """
        Build OR-Tools data dictionary from nodes and travel times.
        """
        nodes = [depot_id] + df_nodes['site_id'].tolist()
        n = len(nodes)

        # Time matrix (minutes)
        time_matrix = [[0]*n for _ in range(n)]
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j:
                    time_matrix[i][j] = 0
                else:
                    row = tt_df[(tt_df['from_site'] == ni) & (tt_df['to_site'] == nj)]
                    time_matrix[i][j] = int(row['travel_minutes'].iloc[0]) if not row.empty else 9999

        # Demands and service times
        demands = [0] + df_nodes['forecast_units'].astype(int).tolist()
        service_times = [0] + df_nodes['service_time_minutes'].fillna(30).astype(int).tolist()

        def to_min(tstr):
            hh, mm = map(int, tstr.split(':'))
            return hh*60 + mm

        # Time windows
        windows = [(0, 24*60)]  # depot open all day
        for _, row in df_nodes.iterrows():
            windows.append((to_min(row['open_time']), to_min(row['close_time'])))

        return {
            'time_matrix': time_matrix,
            'demands': demands,
            'service_times': service_times,
            'time_windows': windows,
            'vehicle_capacities': [barge_capacity_units],
            'num_vehicles': 1,
            'depot': 0,
            'nodes': nodes
        }



    def __solve_cvrptw(self, data):
        """
        Solve CVRPTW problem with OR-Tools and return route as a list of dicts.
        """
        total_demand = sum(data['demands'])
        total_capacity = sum(data['vehicle_capacities'])
        if total_demand > total_capacity:
            print(f"Quick check: total demand {total_demand} exceeds capacity {total_capacity}")

        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        # Transit callback: travel + service time
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['time_matrix'][from_node][to_node]) + int(data['service_times'][from_node])

        transit_idx = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Time dimension
        horizon = 24*60*7
        routing.AddDimension(transit_idx, horizon, horizon, True, 'Time')
        time_dim = routing.GetDimensionOrDie('Time')

        # Apply time windows
        for idx, window in enumerate(data['time_windows']):
            index = manager.NodeToIndex(idx)
            w0, w1 = window
            if w1 < w0:
                w0, w1 = 0, 24*60
            time_dim.CumulVar(index).SetRange(w0, w1)

        # Capacity dimension
        def demand_callback(from_index):
            return int(data['demands'][manager.IndexToNode(from_index)])
        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_idx, 0, data['vehicle_capacities'], True, 'Capacity')

        # Solver parameters
        # Old code:
        # params = pywrapcp.DefaultRoutingSearchParameters()
        # params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        # params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        # params.time_limit.seconds = 30
        # params.log_search = True

        # New code:
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 60
        params.log_search = True

        print("Starting solver...")
        solution = routing.SolveWithParameters(params)

        status_map = {
            0: "ROUTING_NOT_SOLVED",
            1: "ROUTING_SUCCESS",
            2: "ROUTING_FAIL",
            3: "ROUTING_FAIL_TIMEOUT",
            4: "ROUTING_INVALID",
        }

        if solution:
            route = []
            index = routing.Start(0)
            order = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != data['depot']:
                    route.append({'order': order, 'site_id': data['nodes'][node], 'qty': data['demands'][node]})
                    order += 1
                index = solution.Value(routing.NextVar(index))
            print("Solver found a solution.")
            return route
        else:
            print("Solver did not find a solution. Status =", status_map.get(routing.status(), routing.status()))
            return None


    def run(self, week_start='2025-10-13'):
        """
        Run optimizer for a given week_start and return the dispatch route.
        """
        df_nodes, tt_df = self.__build_week_input(week_start)
        if df_nodes.empty or tt_df.empty:
            print("No data to solve for this week.")
            return None

        barge = pd.read_csv(self.barge_file).iloc[0]
        data = self.__create_data_model(df_nodes, tt_df, int(barge['total_capacity_units']))
        sol = self.__solve_cvrptw(data)
        print("Solution route:", sol)
        return sol
