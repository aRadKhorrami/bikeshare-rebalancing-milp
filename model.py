from pyscipopt import Model, quicksum
import data

def solve_model(use_fleet_constraint=False, data_source='sample',
                h=0.1, p=10.0, F=5, M=10000,
                subset_stations=None, subset_times=None,
                time_limit=120):
    """
    Solve the exact MILP from your professor's document.
    """
    # Load data
    if data_source == 'sample':
        S, T, I0, C, D, c, _, _, _, _ = data.get_sample_data()
    else:
        S, T, I0, C, D, c, _, _, _, _ = data.load_real_data(
            '202510-capitalbikeshare-tripdata.csv',
            'Capital_Bikeshare_Locations.csv'
        )

    # Apply subset
    if subset_stations:
        S = [s for s in S if s in subset_stations]
    if subset_times:
        T = [t for t in T if t in subset_times]

    # Filter dictionaries
    I0 = {s: I0[s] for s in S}
    C = {s: C[s] for s in S}
    D = {(i,t): D.get((i,t), 0) for i in S for t in T}
    c = {(i,j): c.get((i,j), 0) for i in S for j in S if i != j}

    model = Model("Bikeshare_Rebalancing")
    model.setParam('limits/time', time_limit)

    # Variables
    f = {(i,j,t): model.addVar(vtype="C", lb=0, name=f"f_{i}_{j}_{t}")
         for i in S for j in S if i != j for t in T}
    I = {(i,t): model.addVar(vtype="C", lb=0, name=f"I_{i}_{t}") for i in S for t in T}
    B = {(i,t): model.addVar(vtype="C", lb=0, name=f"B_{i}_{t}") for i in S for t in T}

    if use_fleet_constraint:
        x = {(i,j,t): model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")
             for i in S for j in S if i != j for t in T}

    # Objective: min Z = Σ c_ij f_ijt + h Σ I_it + p Σ B_it
    obj = (quicksum(c[(i,j)] * f[(i,j,t)] for i in S for j in S if i != j for t in T) +
           h * quicksum(I[(i,t)] for i in S for t in T) +
           p * quicksum(B[(i,t)] for i in S for t in T))
    model.setObjective(obj, "minimize")

    # 6.1 Bike Balance
    min_t = min(T)
    for i in S:
        for t in T:
            prev = I[(i, t-1)] if t > min_t else I0[i]
            inflow = quicksum(f[(j,i,t)] for j in S if j != i)
            outflow = quicksum(f[(i,j,t)] for j in S if j != i)
            model.addCons(I[(i,t)] == prev + inflow - outflow - D[(i,t)] + B[(i,t)],
                          f"balance_{i}_{t}")

    # 6.2 Capacity
    for i in S:
        for t in T:
            model.addCons(I[(i,t)] <= C[i], f"cap_{i}_{t}")

    # 6.3 Non-negativity (already set by lb=0)

    # 6.4 Fleet constraint (optional)
    if use_fleet_constraint:
        for t in T:
            model.addCons(quicksum(x[(i,j,t)] for i in S for j in S if i != j) <= F,
                          f"fleet_{t}")
        for i in S:
            for j in S:
                if i != j:
                    for t in T:
                        model.addCons(f[(i,j,t)] <= M * x[(i,j,t)], f"link_{i}_{j}_{t}")

    # Solve
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