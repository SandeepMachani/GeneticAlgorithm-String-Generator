import numpy as np
import matplotlib.pyplot as plt
import ga 
import math
# cost function

def costFunction(x):
    # print(type(x))
    # v = x[0]**2 + math.sin(x[1])
    # v = (x[0]+ 2*x[1] -7)**2 + (2* x[0] + x[1] -5)**2
    # v = x[0]**2 + x[1]**2
    # v = 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2
    v =-math.cos(x[0])* math.cos(x[1])* math.exp(-((x[0] - math.pi)**2 + (x[1] - math.pi)**2))
    return v

# Problem definition
    # create a dictionary to store problem details
problem = {
    'costFunc' : costFunction,
    'nVar'     : 2,
    'varMin'   : -100,
    'varMax'   : 100
}
# problem.costFunc = costFunction
# problem.nVar = 5
# problem.varMin = -10
# problem.varMax = 10

# GA parameters
npop = [1,2, 20, ]
parameters = {
    'maxIt'  : 100,
    'pc'     : 1,
    'mu'     : 0.1

}


# Run GA
best_sol = []
for npop in npop:
    parameters['npop'] = npop
    out = ga.run(problem, parameters)
    best_sol.append(out['bestsol'])
print(best_sol)
plt.show()
# print(out)
# Results

# import numpy as np
# import matplotlib.pyplot as plt
# import string
# import ga 

# # Target word to guess
# target_word = "GENETIC"

# # Fitness function
# def costFunction(word):
#     cost = 0
#     for i in range(len(target_word)):
#         if word[i] != target_word[i]:
#             cost += 1
#     return cost

# # Problem definition
# problem = {
#     'costFunc': costFunction,
#     'nVar': len(target_word),
#     'varMin': ord('A'),
#     'varMax': ord('Z')
# }

# # GA parameters
# npop = [30]
# parameters = {
#     'maxIt': 1000,
#     'pc': 0.8,
#     'gamma': 0.1,
#     'mu': 0.1,
#     'costFunc': costFunction  # Update parameter name to 'costFunc'
# }

# # Run GA
# best_sols = []
# for npop_size in npop:
#     parameters['npop'] = npop_size
#     out = ga.run(problem, parameters)
#     best_sols.append(out['bestsol'])

#     # Print iteration of words
#     print("Best Word at Each Iteration:")
#     for i, word in enumerate(out['pop']):
#         print(f"Iteration {i+1}: {word['position'].tobytes().decode('latin-1')}")


# # Plot results
# plt.figure()
# for i, npop_size in enumerate(npop):
#     plt.plot(np.arange(parameters['maxIt']), out['bestcost'], label=f"Population Size: {npop_size}")
# plt.xlabel("Generation")
# plt.ylabel("Best Fitness")
# plt.title("Genetic Algorithm - Word Guess")
# plt.legend()
# plt.show()
