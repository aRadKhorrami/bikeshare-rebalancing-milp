"""
Bikeshare Rebalancing Problem: Final MILP Model
===============================================

This module implements the exact Mixed-Integer Linear Programming (MILP) model
as defined in the final project proposal (December 2025) by Ali Rad Khorrami,
incorporating feedback from Professor Lenz.

The model minimizes total transportation, holding, and unmet demand penalty costs
over a discrete time horizon using real Capital Bikeshare data from October 2025.

Mathematical formulation matches Sections 5–6 of the proposal:
- Objective: min Σ c_ij f_ijt + h Σ I_it + p Σ B_it
- Constraints: bike balance, capacity, non-negativity, optional fleet size 
- Optional service level

Supports both **SCIP** and **Gurobi** solvers (Gurobi recommended for large instances).
"""

from pyscipopt import Model as SCIPModel, quicksum
from gurobipy import Model as GurobiModel, GRB
import data

def solve_model(use_fleet_constraint=False, data_source='sample',
                h=0.1, p=10.0, F=5, M=10000,
                subset_stations=None, subset_times=None,
                time_limit=300, gap_limit=0.01, service_level=None,  # e.g., 0.9 for 90% fulfillment (None = disabled) 
                solver="scip"):  
    """
    Solve the bikeshare rebalancing MILP using SCIP or Gurobi.

    Implements constraints 6.1–6.4 exactly as in the final proposal:
        6.1 Bike balance
        6.2 Station capacity
        6.3 Non-negativity
        6.4 (Optional) Fleet-size constraint with big-M linking

    Parameters
    ----------
    use_fleet_constraint : bool, default=False
        If True, limits number of simultaneous truck movements ≤ F using binary x_ijt.
    data_source : {'sample', 'real'}, default='sample'
        Source of input data. 'real' loads October 2025 Capital Bikeshare trip data.
    h : float, default=0.1
        Holding cost per bike per time period ($/bike/period).
    p : float, default=10.0
        Penalty cost per unmet rental demand ($/lost rental).
    F : int, default=5
        Maximum number of trucks available (only used if use_fleet_constraint=True).
    M : int, default=10000
        Big-M constant for logical linking constraints.
    subset_stations : list or None, default=None
        Subset of station names to include (for large-scale testing, e.g., top 100).
    subset_times : list or None, default=None
        Subset of time periods to solve (e.g., first 12 two-hour periods).
    time_limit : int, default=300
        Maximum solving time in seconds.
    service_level : float or None, default=None
            Minimum demand fulfillment fraction (e.g., 0.9 for 90%). 
            Adds constraint B_{i,t} ≤ (1 - service_level) × D_{i,t} (from PDF Section 8).
            If None, no constraint is added.        
    solver : {'scip', 'gurobi'}, default='scip'
        MILP solver to use. Gurobi is significantly faster for large instances.

    Returns
    -------
    results : dict
        Dictionary containing:
            - 'f': {(i,j,t): value} – number of bikes transported
            - 'I': {(i,t): value} – inventory at end of period
            - 'B': {(i,t): value} – unmet demand (slack)
            - 'x': {(i,j,t): 0/1} – truck usage (if fleet constraint active)
            - 'obj_val': float – optimal (or best found) objective value
    status : str
        Solver status: 'optimal', 'timelimit', 'infeasible', etc.

    Notes
    -----
    - Station names are used as keys (string-based indexing).
    - Euclidean distance used as transportation cost c_ij.
    - Initial inventory I0 set to ~50% of capacity.

    See Also
    --------
    data.load_real_data : Loads and processes real October 2025 data.
    """
    # Load parameters (S, T, I0, C, D, c, coords)
    if data_source == 'sample':
        S, T, I0, C, D, c, _, _, _, _ = data.get_sample_data()
        coords = None  # For sample: No coordinates available
    else:
        S, T, I0, C, D, c, _, _, _, _, coords = data.load_real_data( 
            '202510-capitalbikeshare-tripdata.csv',
            'Capital_Bikeshare_Locations.csv'
        )

    if subset_stations:
        S = [s for s in S if s in subset_stations]
        if coords:
            coords = {s: coords[s] for s in S}  # Subset coordinates accordingly        
    if subset_times:
        T = [t for t in T if t in subset_times]

    I0 = {s: I0[s] for s in S}
    C = {s: C[s] for s in S}
    D = {(i,t): D.get((i,t), 0) for i in S for t in T}
    c = {(i,j): c.get((i,j), 0) for i in S for j in S if i != j}

    # ========================
    # SETUP AND SOLVE MODEL
    # ========================
    # Branch by solver; define variables, objective, constraints.
    if solver.lower() == "gurobi":
        model = GurobiModel("Bikeshare_Rebalancing_Gurobi")
        model.setParam('TimeLimit', time_limit)
        model.setParam('OutputFlag', 1)  # 0 = Suppress Gurobi log (optional)
        if gap_limit > 0:
            model.setParam('MIPGap', gap_limit)  # Set gap tolerance for Gurobi

        # Variables (Gurobi syntax)
        f = {(i,j,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{t}")
             for i in S for j in S if i != j for t in T}
        I = {(i,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{t}") for i in S for t in T}
        B = {(i,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"B_{i}_{t}") for i in S for t in T}

        if use_fleet_constraint:
            x = {(i,j,t): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
                 for i in S for j in S if i != j for t in T}

        # Objective: Minimize total cost (transport + holding + penalty)
        obj = (sum(c[(i,j)] * f[(i,j,t)] for i in S for j in S if i != j for t in T) +
               h * sum(I[(i,t)] for i in S for t in T) +
               p * sum(B[(i,t)] for i in S for t in T))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        min_t = min(T)
        # Bike balance (6.1): I_{i,t} = prev + in - out - D + B
        for i in S:
            for t in T:
                prev = I[(i, t-1)] if t > min_t else I0[i]
                inflow = sum(f[(j,i,t)] for j in S if j != i)
                outflow = sum(f[(i,j,t)] for j in S if j != i)
                model.addConstr(I[(i,t)] == prev + inflow - outflow - D[(i,t)] + B[(i,t)],
                                name=f"balance_{i}_{t}")

        # Capacity (6.2): I <= C
        for i in S:
            for t in T:
                model.addConstr(I[(i,t)] <= C[i], name=f"cap_{i}_{t}")

        # Optional fleet (6.4): sum x <= F, f <= M x
        if use_fleet_constraint:
            for t in T:
                model.addConstr(sum(x[(i,j,t)] for i in S for j in S if i != j) <= F,
                                name=f"fleet_{t}")
            for i in S:
                for j in S:
                    if i != j:
                        for t in T:
                            model.addConstr(f[(i,j,t)] <= M * x[(i,j,t)], name=f"link_{i}_{j}_{t}")

        # Service-level constraint from PDF Section 8: B_i,t <= 0.1 * D_i,t (90% fulfillment per station-period)
        if service_level is not None:
                max_unmet_fraction = 1.0 - service_level  # e.g., 0.1 for 90%
                for i in S:
                    for t in T:
                        model.addConstr(B[(i,t)] <= max_unmet_fraction * D[(i,t)],
                                    name=f"service_{i}_{t}")

        # Solve the model
        model.optimize()
        # Map Gurobi status codes to strings
        status = "optimal" if model.Status == GRB.OPTIMAL else \
                 "timelimit" if model.Status == GRB.TIME_LIMIT else "infeasible"

        if model.SolCount > 0:
            obj_val = model.ObjVal
            results = {
                'f': {(i,j,t): f[(i,j,t)].X for i in S for j in S if i != j for t in T},
                'I': {(i,t): I[(i,t)].X for i in S for t in T},
                'B': {(i,t): B[(i,t)].X for i in S for t in T},
                'obj_val': obj_val  # Note: 'is_optimal' and 'gap' not directly from Gurobi here; add if needed
            }
            if use_fleet_constraint:
                results['x'] = {(i,j,t): x[(i,j,t)].X for i in S for j in S if i != j for t in T}
            results['coords'] = coords  # NEW: Add to results
            return results, status
        else:
            return None, status

    else:  # SCIP solver branch
        model = SCIPModel("Bikeshare_Rebalancing")
        model.setParam('limits/time', time_limit)

        # To get the best solution on timeout
        model.setParam('limits/gap', gap_limit)  # User's gap limit (e.g., 0.01 for 1%)
        model.setParam('limits/absgap', 0.0)

        f = {(i,j,t): model.addVar(vtype="C", lb=0, name=f"f_{i}_{j}_{t}")
             for i in S for j in S if i != j for t in T}
        I = {(i,t): model.addVar(vtype="C", lb=0, name=f"I_{i}_{t}") for i in S for t in T}
        B = {(i,t): model.addVar(vtype="C", lb=0, name=f"B_{i}_{t}") for i in S for t in T}

        if use_fleet_constraint:
            x = {(i,j,t): model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")
                 for i in S for j in S if i != j for t in T}

        obj = (quicksum(c[(i,j)] * f[(i,j,t)] for i in S for j in S if i != j for t in T) +
               h * quicksum(I[(i,t)] for i in S for t in T) +
               p * quicksum(B[(i,t)] for i in S for t in T))
        model.setObjective(obj, "minimize")

        min_t = min(T)
        for i in S:
            for t in T:
                prev = I[(i, t-1)] if t > min_t else I0[i]
                inflow = quicksum(f[(j,i,t)] for j in S if j != i)
                outflow = quicksum(f[(i,j,t)] for j in S if j != i)
                model.addCons(I[(i,t)] == prev + inflow - outflow - D[(i,t)] + B[(i,t)],
                              f"balance_{i}_{t}")

        for i in S:
            for t in T:
                model.addCons(I[(i,t)] <= C[i], f"cap_{i}_{t}")

        if use_fleet_constraint:
            for t in T:
                model.addCons(quicksum(x[(i,j,t)] for i in S for j in S if i != j) <= F,
                              f"fleet_{t}")
            for i in S:
                for j in S: 
                    if i != j:
                        for t in T:
                            model.addCons(f[(i,j,t)] <= M * x[(i,j,t)], f"link_{i}_{j}_{t}")
        
        # Service-level constraint from PDF Section 8: B_i,t <= 0.1 * D_i,t (90% fulfillment per station-period)
        if service_level is not None:
                max_unmet_fraction = 1.0 - service_level  # e.g., 0.1 for 90%
                for i in S:
                    for t in T:
                        model.addCons(B[(i,t)] <= max_unmet_fraction * D[(i,t)],
                                    name=f"service_{i}_{t}")

        # Solve with SCIP
        model.optimize()
        status = model.getStatus()

        if status in ["optimal", "timelimit", "gaplimit", "userinterrupt"]:  # Handle feasible statuses (may have solution)
            try:
                # Try to get the best found solution
                if model.getNSols() > 0:  # If at least one solution was found
                    obj_val = model.getObjVal()  # Best found solution
                    is_optimal = (status == "optimal")
                    
                    # Get gap for both timelimit AND gap limit reached
                    if status in ["timelimit", "gaplimit"]:  # Compute relative gap for non-optimal cases
                        try:
                            gap_value = model.getGap()
                            if gap_value is None:
                                gap_value = 0.0
                        except:
                            gap_value = 0.0
                    else:
                        gap_value = 0.0
                else:
                    obj_val = None
                    is_optimal = False
                    gap_value = 0.0
                        
                if obj_val is not None:
                    results = {
                        'f': {(i,j,t): model.getVal(f[(i,j,t)]) 
                              for i in S for j in S if i != j for t in T},
                        'I': {(i,t): model.getVal(I[(i,t)]) for i in S for t in T},
                        'B': {(i,t): model.getVal(B[(i,t)]) for i in S for t in T},
                        'obj_val': obj_val,
                        'is_optimal': is_optimal,  # Include solution quality metrics
                        'gap': gap_value
                    }
                    if use_fleet_constraint:
                        results['x'] = {(i,j,t): model.getVal(x[(i,j,t)]) 
                                        for i in S for j in S if i != j for t in T}
                    results['coords'] = coords  # NEW: Add to results
                    return results, status
                else:
                    return None, "no_solution_found"
                    
            except Exception as e:
                # If there's an error extracting values
                print(f"Warning: Could not extract solution values: {e}")
                return None, f"error_extracting_solution: {str(e)}"
        else:
            return None, status