import sys
import numpy as np
import cvxpy as cp
import random
import csv

# ---------------------------------------------------------------------------------------------------------------
# branch and bound algorithm adopted from 15780 course work
from heapq import heappush, heappop

class PQNode:
    '''
    Represents the solution to a problem relaxation.
    '''
    def __init__(self, prob_value, var_value, constraints):
        self.prob_value = prob_value
        self.var_value = var_value
        self.constraints = constraints

    def __gt__(self, other_node):
        return self.prob_value > other_node.prob_value

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def size(self):
        return len(self.elements)

    def push(self, element, priority):
        '''
        Adds an element along with its priority.
        '''
        heappush(self.elements, (priority, element))

    def pop(self):
        '''
        Retrieves the element with lowest priority.
        '''
        return heappop(self.elements)[1]

def check_integer_value(x):
    for i in range(len(x)):
        if not np.isclose(x[i].round(), x[i], atol=1e-7):
            return i, x[i]
    return -1, 0

def solve_relaxation(var1, var2, var3, objective, constraints, new_constraints):
    problem = cp.Problem(objective, constraints + new_constraints)
    problem.solve(solver='ECOS')
    if (var1.value is None) or (var2.value is None) or (var3.value is None):
        return PQNode(problem.value, None, new_constraints)
    final_var_value = [0 for i in range(len(var1.value[0]) + len(var2.value[0]) + len(var3.value[0]))]
    for i in range(len(var1.value[0])):
        final_var_value[i] = var1.value[0][i]
    for i in range(len(var2.value[0])):
        final_var_value[len(var1.value[0]) + i] = var2.value[0][i]
    for i in range(len(var3.value[0])):
        final_var_value[len(var1.value[0]) + len(var2.value[0]) + i] = var3.value[0][i] 
    return PQNode(problem.value, final_var_value, new_constraints)
    
def branch_and_bound(var1, var2, var3, objective, constraints, l1, l2):
    pq = PriorityQueue()
    cur_node = solve_relaxation(var1, var2, var3, objective, constraints, [])
    pq.push(cur_node, cur_node.prob_value)
    while pq.size() > 0:
        cur_node = pq.pop()
        non_int_index, non_int_val = check_integer_value(cur_node.var_value)
        if non_int_index == -1:
            return cur_node.prob_value, cur_node.var_value
        lower = int(non_int_val) if non_int_val > 0 else (int(non_int_val) - 1)
        upper = lower + 1
        if non_int_index < l1:
            new_constraints_1 = cur_node.constraints + [var1[0][non_int_index] <= lower]
        elif non_int_index < l1 + l2:
            new_constraints_1 = cur_node.constraints + [var2[0][non_int_index - l1] <= lower]
        else:
            new_constraints_1 = cur_node.constraints + [var3[0][non_int_index - l1 - l2] <= lower]
        new_node_1 = solve_relaxation(var1, var2, var3, objective, constraints, new_constraints_1)
        if not (new_node_1.var_value is None):
            pq.push(new_node_1, new_node_1.prob_value)
        if non_int_index < l1:
            new_constraints_2 = cur_node.constraints + [var1[0][non_int_index] >= upper]
        elif non_int_index < l1 + l2:
            new_constraints_2 = cur_node.constraints + [var2[0][non_int_index - l1] >= upper]
        else:
            new_constraints_2 = cur_node.constraints + [var3[0][non_int_index - l1 - l2] >= upper]
        new_node_2 = solve_relaxation(var1, var2, var3, objective, constraints, new_constraints_2)
        if not (new_node_2.var_value is None):
            pq.push(new_node_2, new_node_2.prob_value)
    return None
#----------------------------------------------------------------------------------------------------------------

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

def solve_lp(num_images, num_categories, budget, num_of_each_species, matrix, costs):
    alloc_all = cp.Variable((1, num_images * num_categories), integer = True)
    alloc_images = cp.Variable((1, num_images), integer = True)
    alloc_categories = cp.Variable((1, num_categories), integer = True)
    # alloc_all = cp.Variable((1, num_images * num_categories))
    # alloc_images = cp.Variable((1, num_images))
    # alloc_categories = cp.Variable((1, num_categories))

    # constraints
    constraints = []
    c_image = np.ones(num_images)
    constraints.append(alloc_images @ costs.T <= budget)

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
    objective = cp.Maximize(cp.sum(alloc_categories))

    # budget constraint
    problem = cp.Problem(objective, constraints)
    # problem.solve(solver='ECOS')
    problem.solve(solver='GLPK_MI')

    return problem.value, alloc_all.value, alloc_images.value
    # prob_value, var_value = branch_and_bound(alloc_images, alloc_categories, alloc_all, objective, constraints, num_images, num_categories)
    # return prob_value, var_value


# check if budget can cover at least one location
def check_budget(costs, budget):
    for cost in costs:
        if budget >= cost:
            return True
    return False


# randomly select locations to preserve
def random_baseline(matrix, costs, num_images, num_categories, budget, num_of_each_species):
    indices = list(range(num_images))
    # print(indices)
    result = np.zeros(num_categories)

    while check_budget(costs, budget):
        selected_idx = random.sample(indices, k=1)[0]
        # print(selected_idx)
        budget -= costs[selected_idx]
        indices.remove(selected_idx)
        # print(indices)
        result += matrix[selected_idx]

    # print(result)
    for i in range(num_categories):
        result[i] = 1 if result[i] >= num_of_each_species else 0

    return np.sum(result)


def greedy_baseline(matrix, costs, num_images, num_categories, budget, num_of_each_species):
    sums = np.sum(matrix, axis = 0)
    max_species = np.argmax(sums)
    
    result = np.zeros(num_categories)
    i = 0
    while budget > 0:
        max_probs = matrix[:, max_species]
        img_argmax = np.argmax(max_probs)
        result += matrix[img_argmax]
        budget -= costs[img_argmax]
        matrix = np.delete(matrix, img_argmax, 0)
        costs = np.delete(costs, img_argmax)

    print(result)
    for i in range(num_categories):
        result[i] = 1 if result[i] >= num_of_each_species else 0

    return np.sum(result)


# input: csv file, n = # of lines to process
# output: np array
def process_csv(filename, n):
    result = []
    with open(filename) as file:
        reader = csv.reader(file)
        counter = 0
        for row in reader:
            if counter > n:
                break
            result.append(row)
            counter += 1
    file.close()
    result = np.array(result, dtype=float)
    return result

def uniform_costs(n):
    costs = np.ones(n, dtype=float)
    return costs

if __name__ == '__main__':
    num_images = 15
    num_categories = 10
    budget = 5
    num_of_each_species = 1
    matrix = process_csv('sample_predictions.csv', num_images)

    # costs = np.zeros(num_images)
    # for i in range(num_images):
    #     costs[i] = 1.0
    costs = uniform_costs(num_images)

    prob_val, alloc_val, alloc_img_val = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix, costs)
    # print(prob_val)
    # print(alloc_img_val)
    print("lp result = {}".format(prob_val))

    # random baseline
    random_res = random_baseline(matrix, costs, num_images, num_categories, budget, num_of_each_species)
    print("random selection result = {}".format(random_res))

    # greedy baseline
    greedy_result = greedy_baseline(np.copy(matrix), costs, num_images, num_categories, budget, num_of_each_species)
    print("new greedy selection result = {}".format(greedy_result))
