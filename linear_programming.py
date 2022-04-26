import sys
import numpy as np
import cvxpy as cp
import random
import csv

from heapq import heappush, heappop

'''
You can use the following classes in your algorithm (although you are not required to).
'''

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
        # l1 = len(var1.value[0])
        # l2 = len(var2.value[0])
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
    # alloc_all = cp.Variable((1, num_images * num_categories))
    # alloc_images = cp.Variable((1, num_images))
    # alloc_categories = cp.Variable((1, num_categories))

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

    # pv, vv = branch_and_bound(alloc_images, alloc_categories, alloc_all, objective, constraints, num_images, num_categories)
    # return pv, vv

    return problem.value, alloc_all.value, alloc_images.value



if __name__ == '__main__':
    # filename = sys.argv[1]
    # budget = sys.argv[2]
    # num_images = 10
    # num_categories = 4
    # budget = 4
    # num_of_each_species = 2
    # matrix = np.zeros((num_images, num_categories))
    # for i in range(0, 2):
    #     matrix[i][0] = 0.25
    #     matrix[i][1] = 0.25
    #     matrix[i][2] = 0.25
    #     matrix[i][3] = 0.25
    # for i in range(2, 5):
    #     matrix[i][0] = 0.5
    #     matrix[i][1] = 0.5
    # for i in range(5, 10):
    #     matrix[i][2] = 0.5
    #     matrix[i][3] = 0.5

    # print(matrix)

    # prob_val, alloc_val, alloc_img_val = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix)
    # print(prob_val)
    # print(alloc_val)
    # print(alloc_img_val)

    # num_images = 50
    # num_categories = 10
    # budget = 10
    # num_of_each_species = 1
    # matrix = np.zeros((num_images, num_categories))
    # for i in range(0, num_images):
    #     random_sample = np.zeros(num_categories)
    #     for j in range(num_categories):
    #         random_sample[j] = random.random()
    #     random_sample = random_sample / np.sum(random_sample)
    #     # print(np.sum(random_sample))
    #     matrix[i] = random_sample

    # # print(matrix)

    # prob_val, alloc_val, alloc_img_val = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix)
    # # pv, vv = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix)
    # # print(pv)
    # # print(vv[:50])

    # print(prob_val)
    # # print(alloc_val)
    # print(alloc_img_val)
    num_images = 15
    matrix = []
    with open('sample_predictions.csv') as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if count > num_images:
                break
            matrix.append(row)
            count += 1
    file.close()
    matrix = np.array(matrix, dtype=float)
    
    
    num_categories = 10
    budget = 5
    num_of_each_species = 1

    prob_val, alloc_val, alloc_img_val = solve_lp(num_images, num_categories, budget, num_of_each_species, matrix)
    # print(prob_val)
    # print(alloc_img_val)
    print("lp result = {}".format(prob_val))

    # random selection of budget number of images
    indices = random.sample(list(range(num_images)), k=budget)
    print(indices)
    res = np.zeros(num_categories)
    for i in indices:
        # for j in range(num_categories):
        #     res[j] = matrix[i][j]
        res = res + matrix[i]
    print(res)

    for j in range(num_categories):
        if res[j] >= num_of_each_species:
            res[j] = 1
        else:
            res[j] = 0

    print("random selection result = {}".format(np.sum(res)))

    # greedy 
    sums = np.sum(matrix, axis = 0)
    argmax = np.argmax(sums)
    
    res = np.zeros(num_categories)
    for i in range(budget):
        max_probs = matrix[:, argmax]
        img_argmax = np.argmax(max_probs)
        res = res + matrix[img_argmax]
        matrix = np.delete(matrix, img_argmax, 0)

    print(res)

    for j in range(num_categories):
        if res[j] >= num_of_each_species:
            res[j] = 1
        else:
            res[j] = 0

    print("greedy selection result = {}".format(np.sum(res)))

