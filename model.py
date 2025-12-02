# model.py
from pyscipopt import Model as SCIPModel, quicksum
from gurobipy import Model as GurobiModel, GRB
import data

def solve_model(use_fleet_constraint=False, data_source='sample',
                h=0.1, p=10.0, F=5, M=10000,
                subset_stations=None, subset_times=None,
                time_limit=120, solver="scip"):  # ← NEW PARAMETER
    """
    Solve the exact MILP model using either SCIP or Gurobi.
    """
    # Load data (same as before)
    if data_source == 'sample':
        S, T, I0, C, D, c, _, _, _, _ = data.get_sample_data()
    else:
        S, T, I0, C, D, c, _, _, _, _ = data.load_real_data(
            '202510-capitalbikeshare-tripdata.csv',
            'Capital_Bikeshare_Locations.csv'
        )

    if subset_stations:
        S = [s for s in S if s in subset_stations]
    if subset_times:
        T = [t for t in T if t in subset_times]

    I0 = {s: I0[s] for s in S}
    C = {s: C[s] for s in S}
    D = {(i,t): D.get((i,t), 0) for i in S for t in T}
    c = {(i,j): c.get((i,j), 0) for i in S for j in S if i != j}

    # ========================
    # SOLVER-SPECIFIC SETUP
    # ========================
    if solver.lower() == "gurobi":
        model = GurobiModel("Bikeshare_Rebalancing_Gurobi")
        model.setParam('TimeLimit', time_limit)
        model.setParam('OutputFlag', 0)  # Suppress Gurobi log (optional)
        # model.setParam('MIPGap', 0.01)  # Optional: set gap tolerance

        # Variables (Gurobi syntax)
        f = {(i,j,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{t}")
             for i in S for j in S if i != j for t in T}
        I = {(i,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{t}") for i in S for t in T}
        B = {(i,t): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"B_{i}_{t}") for i in S for t in T}

        if use_fleet_constraint:
            x = {(i,j,t): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
                 for i in S for j in S if i != j for t in T}

        # Objective
        obj = (sum(c[(i,j)] * f[(i,j,t)] for i in S for j in S if i != j for t in T) +
               h * sum(I[(i,t)] for i in S for t in T) +
               p * sum(B[(i,t)] for i in S for t in T))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        min_t = min(T)
        for i in S:
            for t in T:
                prev = I[(i, t-1)] if t > min_t else I0[i]
                inflow = sum(f[(j,i,t)] for j in S if j != i)
                outflow = sum(f[(i,j,t)] for j in S if j != i)
                model.addConstr(I[(i,t)] == prev + inflow - outflow - D[(i,t)] + B[(i,t)],
                                name=f"balance_{i}_{t}")

        for i in S:
            for t in T:
                model.addConstr(I[(i,t)] <= C[i], name=f"cap_{i}_{t}")

        if use_fleet_constraint:
            for t in T:
                model.addConstr(sum(x[(i,j,t)] for i in S for j in S if i != j) <= F,
                                name=f"fleet_{t}")
            for i in S:
                for j in S:
                    if i != j:
                        for t in T:
                            model.addConstr(f[(i,j,t)] <= M * x[(i,j,t)], name=f"link_{i}_{j}_{t}")

        # Optimize
        model.optimize()
        status = "optimal" if model.Status == GRB.OPTIMAL else \
                 "timelimit" if model.Status == GRB.TIME_LIMIT else "infeasible"

        if model.SolCount > 0:
            obj_val = model.ObjVal
            results = {
                'f': {(i,j,t): f[(i,j,t)].X for i in S for j in S if i != j for t in T},
                'I': {(i,t): I[(i,t)].X for i in S for t in T},
                'B': {(i,t): B[(i,t)].X for i in S for t in T},
                'obj_val': obj_val
            }
            if use_fleet_constraint:
                results['x'] = {(i,j,t): x[(i,j,t)].X for i in S for j in S if i != j for t in T}
            return results, status
        else:
            return None, status

    else:  # Default: SCIP (original code)
        model = SCIPModel("Bikeshare_Rebalancing")
        model.setParam('limits/time', time_limit)

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

        model.optimize()
        status = model.getStatus()

        if status in ["optimal", "timelimit"]:
            obj_val = model.getObjVal() if status == "optimal" else "timeout"
            results = {
                'f': {(i,j,t): model.getVal(f[(i,j,t)]) for i in S for j in S if i != j for t in T},
                'I': {(i,t): model.getVal(I[(i,t)]) for i in S for t in T},
                'B': {(i,t): model.getVal(B[(i,t)]) for i in S for t in T},
                'obj_val': obj_val
            }
            if use_fleet_constraint:
                results['x'] = {(i,j,t): model.getVal(x[(i,j,t)]) for i in S for j in S if i != j for t in T}
            return results, status
        else:
            return None, status