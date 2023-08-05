import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
def run(problem, parameters):
    
    # Problem information
    costFunc = problem['costFunc']
    nVar = problem['nVar']
    varMin = problem['varMin']
    varMax = problem['varMax']

    # Parameters information 
    maxIt = parameters['maxIt']
    npop = parameters['npop']
    pc = parameters['pc']
    nc = np.round(pc*npop/2)*2
    # print('nc',nc//2)
    mu = parameters['mu']

    # Empty individual (Dictionary)
    empty_individual = {
        'position' : np.empty(nVar),
        'cost' : np.empty(nVar)

    }

    #Best solution ever found
    bestsol = deepcopy(empty_individual)
    bestsol['cost'] = np.inf 

    
    # Initialize population
    pop = [None] * npop

    for i in range (npop):
        pop[i] = deepcopy(empty_individual)
    # pop = empty_individual.deepcopy()
    for i in range(npop):
        pop[i]['position'] = np.random.uniform(varMin, varMax, nVar)
        # print(pop[i]['position'])
        pop[i]['cost'] = costFunc(pop[i]['position'])
        if pop[i]['cost'] < bestsol['cost']:
            bestsol = deepcopy(pop[i])
    
    # BEst cost of an iteration
    bestcost = np.empty(maxIt)


    # Main loop
    for i in range(maxIt):

        # create empty children list
        popc = []
        
        # fill the children list with children
        for _ in range(int(nc//2)): # nc - number of children

            q = np.random.permutation(npop) # create a random list of number till npop
            #select parents
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # create children with crossover

            c1, c2 = crossover(p1, p2)

            # perform mutation

            c1 = mutate(c1, mu)
            c2 = mutate(c2, mu)

            # Applying bounds
            applybounds(c1, varMin, varMax)
            applybounds(c2, varMin, varMax)

            # evaluate first offspring cost
            c1['cost'] = costFunc(c1['position'])
            if c1['cost'] < bestsol['cost']:
                bestsol = deepcopy(c1)

            # evaluate second offspring cost
            c2['cost'] = costFunc(c2['position'])
            if c2['cost'] < bestsol['cost']:
                bestsol = deepcopy(c2)

            # Add offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge , sort and select

        pop += popc
        pop = sorted(pop, key = lambda x: x['cost'])
        pop = pop[0 : npop]

        # store best cost after iteration

        bestcost[i] = bestsol['cost']

        # show iteratyion information
        print('iteration {}: Best Cost {}: Population size {}'.format(i, bestcost[i], str(npop))) 
    
    plt.plot(np.arange(1, i+2), bestcost, label=str(npop))
    plt.xlabel("Generations")
    plt.ylabel("Cost")
    plt.title("Cost vs. Generations with diff population sizes(Beam)")
    plt.legend()
    print(bestsol)
    # plt.show()


    #store the above output 
    out = {
        'pop' : pop,
        'bestsol' : bestsol,
        'bestcost' : bestcost
    }
    return out

def crossover(p1, p2):

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    alpha = np.random.uniform(0, 1, *c1['position'].shape)
    c1['position'] = alpha*p1['position'] + (1-alpha)* p2['position']
    c2['position'] = alpha*p2['position'] + (1-alpha)* p1['position']
    return c1, c2

def mutate(x, mu):
    y = deepcopy(x)
    # print("ran",x)
    # print(type(x))
    mask = np.random.rand(*x['position'].shape) <= mu
    # print(mask)
    ind = np.argwhere(mask)
    # print(ind)
    # print(y)
    y['position'][ind] += np.random.randn(*ind.shape)
    return y

def applybounds(x, varMin, varMax):
    x['position'] = np.maximum(x['position'], varMin)
    x['position'] = np.minimum(x['position'], varMax)
