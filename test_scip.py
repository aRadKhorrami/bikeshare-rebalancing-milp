from pyscipopt import Model
model = Model("test")
x = model.addVar("x")
model.setObjective(x, "minimize")
model.optimize()
print("SCIP WORKS! Status:", model.getStatus())