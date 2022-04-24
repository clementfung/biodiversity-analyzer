import sys
import numpy as np
import cvxpy as cp

# contruct a matrix like:
# 1 0 0
# 1 0 0
# 0 1 0
# 0 1 0
# 0 0 1
# 0 0 1
def construct_A(num_images, num_categories):
    A = np.zeros((num_images * num_categories, num_images))
    for j in range(num_images):
        for i in range(j*num_categories, (j+1)*num_categories):
            A[i][j] = 1
    return A

# contruct a matrix like:
# 1 0 0
# 0 1 0
# 0 0 1
# 1 0 0
# 0 1 0
# 0 0 1
# 1 0 0
# 0 1 0
# 0 0 1
def construct_B(num_images, num_categories, matrix):
    B = np.zeros((num_images * num_categories, num_categories))
    for j in range(num_categories):
        for i in range(j, num_images * num_categories, num_categories):
            B[i][j] = matrix[i//num_categories][j]
    return B

def solve_lp(num_images, num_categories, budget, num_of_each_species, matrix):
    alloc_all = cp.Variable((1, num_images * num_categories), integer = True)
    alloc_images = cp.Variable((1, num_images), integer = True)
    alloc_categories = cp.Variable((1, num_categories), integer = True)

    # constraints
    constraints = []
    c_image = np.ones(num_images)
    constraints.append(alloc_images @ c_image.T <= budget)

    A = construct_A(num_images, num_categories)
    constraints.append(alloc_all @ A <= alloc_images * num_categories)

    # c_species = [num_of_each_species] @ alloc_images
    B = construct_B(num_images, num_categories, matrix)
    constraints.append(alloc_all @ B >= alloc_categories * num_of_each_species)

    constraints.append(alloc_all >= 0)
    constraints.append(alloc_all <= 1)
    constraints.append(alloc_images >= 0)
    constraints.append(alloc_images <= 1)
    constraints.append(alloc_categories >= 0)
    constraints.append(alloc_categories <= 1)

    # objective
    c_categories = np.ones(num_categories)
    # objective = cp.Maximize(cp.sum(alloc_all @ B))
    objective = cp.Maximize(cp.sum(alloc_categories))

    # budget constraint
    problem = cp.Problem(objective, constraints)
    # problem.solve(solver='ECOS')
    problem.solve(solver='GLPK_MI')
    return problem.value, alloc_all.value, alloc_images.value

if __name__ == '__main__':
    # filename = sys.argv[1]
    # budget = sys.argv[2]
    num_images = 10
    num_categories = 4
    budget = 5
    num_of_each_species = 2
    matrix = np.zeros((num_images, num_categories))
    for i in range(0, 2):
        matrix[i][0] = 0.25
        matrix[i][1] = 0.25
        matrix[i][2] = 0.25
        matrix[i][3] = 0.25
    for i in range(2, 5):
        matrix[i][0] = 0.5
        matrix[i][1] = 0.5
    for i in range(5, 10):
        matrix[i][2] = 0.5
        matrix[i][3] = 0.5

    print(matrix)

    prob_val, alloc_val, alloc_img_val = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix)
    print(prob_val)
    print(alloc_val)
    print(alloc_img_val)
