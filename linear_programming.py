import sys
import numpy as np
import cvxpy as cp

def solve_lp(num_images, budget):
    var = cp.Variable(num_images)
    objective = cp.Maximize()
    # budget constraint
    c = np.ones(num_images)
    budget_constraint = [var @ c <= budget]
    constraints = []
    cp.Problem(objective, constraints)
    problem.solve(solver='ECOS')
    return problem.value, var.value

if __name__ == '__main__':
    filename = sys.argv[1]
    budget = sys.argv[2]
    # TODO: construct and solve lp
    pass