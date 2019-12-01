###################################
import numpy as np
import pandas as pd
import GA1
import time
import matplotlib.pyplot as plt
import Output_plot1
from sklearn.metrics import r2_score
###################################
###################################

start = time.time()
np.set_printoptions(precision=6)

data = pd.read_csv('Actual.csv')#, encoding = "utf-8", error_bad_lines=False)
#data_X = np.array(data[['Cons','CAL', 'GR', 'RHOB', 'NPHI', 'LLM', 'DT']])
data_X = np.array(data[["Depth (m)","GR","RHOB","NPHI","RESD","DT","DTs"]])
#print(data_X)


x = data_X
x = np.array(x)
y = np.power(x, 2)
d = np.power(x, 3)
e = np.power(x, 4)
data_X = np.hstack((x, y, d, e))
mean = np.mean(data_X, axis=0)
data_X = (data_X - mean)/np.max(data_X, axis=0)
# print(data["GR"])
# print(data.max())
# print(data.min())

m = 2
Rho_fluid = 1
Rho_ma = 2.44 # Rho_limestone

# Calculate Target Fracture Poroity using Elkewidy and Tiab method
def porosity(Rho_ma):
    data['PHID'] = (Rho_ma - data['RHOB']) / (Rho_ma - Rho_fluid)
    data['PHIt'] = (data['NPHI'] + data['PHID']) / 2
    data['PHIF'] = ((data['PHIt']**(m + 1)) - (data['PHIt']**m)) / ((data['PHIt']**m) - 1)
    return data[['PHIF']]
FPoro = porosity(Rho_ma)
# print(FPoro)

# Data splitting
total_data = 375
train = int(total_data*0.6)
validation = int(total_data*0.8)
test = int(total_data)

# Training data
# Use 60% of total data points as training and rest of the points as test set.
train_data = data_X[0:train]
DEEP1 = np.array(data['Porosity'])
#print(DEEP1)
TEPD = np.vstack(DEEP1[0:train])
PHIF = np.array(FPoro)
Train_Tiab_PHIF = np.vstack(PHIF[0:train])
# print('Train :',"\n",TEPD[0])

# validation data
validation_data = data_X[train:validation]
DEEP3 = np.array(data['Porosity'])
#print(DEEP3[train:validation])
Depth = np.vstack(DEEP3[train:validation])
PHIF = np.array(FPoro)
Validation_Tiab_PHIF = np.vstack(PHIF[train:validation])
# print('Train :',"\n",Depth[0])

# Testing data
test_data = data_X[validation:test]
DEEP2 = np.array(data['Porosity'])
DEPT = np.vstack(DEEP2[validation:test])
Test_Tiab_PHIF = np.vstack(PHIF[validation:test])
# print('Train :',"\n",DEPT[0])

#  Create a starting random population
sol_per_pop = 400
num_parents = 200
num_weights = 28
# print(num_parents)

population = GA1.create_starting_population(sol_per_pop, num_weights)
# print('Random Population :', "\n", population[0:5])

### Adjust the NO. of Generation
num_generations = 502 # adjust plot x axis limits
best_fit = []
iteration = []
train_Cost = []
validation_Cost = []
test_Cost = []

for generation in range(num_generations):

    # Measing the GA porosity using objrctive function
    GA_PHIF = GA1.calculate_Obj_function(train_data, population, num_weights)
    # Measing the fitness of each chromosome in the population.
    fitness = GA1.calculate_fitness(GA_PHIF, Train_Tiab_PHIF,train)

    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    # Following codes were used to take best result while programe is running
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

    # The best result in the current iteration.
    print("Generation:",generation,'  | Fitness:', np.min(fitness)) ### error
    best_match_idx = np.where(fitness == np.min(fitness))
    print(population[best_match_idx, :])

    best_fit.append(np.min(fitness))
    iteration.append(generation)

    train_cost_plot = GA1.cost(Train_Tiab_PHIF,population, best_match_idx,train, train_data, num_weights)
    train_Cost.append(train_cost_plot)
    validation_cost_plot = GA1.cost(Validation_Tiab_PHIF,population, best_match_idx,validation, validation_data, num_weights)
    validation_Cost.append(validation_cost_plot)
    test_cost_plot = GA1.cost(Test_Tiab_PHIF,population, best_match_idx,test, test_data, num_weights)
    test_Cost.append(test_cost_plot)

    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

    # Selecting the best parents in the population for mating.
    parents = GA1.select_mating_pool(population, fitness, num_parents)
    # Generating next generation using crossover.
    offspring = GA1.breed_by_crossover(parents, Train_Tiab_PHIF, train_data, offspring_size=(population.shape[0] - parents.shape[0], num_weights))
    # Generating mutated offspring.
    offspring_mutation = GA1.randomly_mutate_population(offspring)
    # Creating the new population based on the parents and offspring.
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = offspring_mutation


bestfit = np.array(best_fit)
gen = np.array(iteration)
bestcost_train = np.array(train_Cost)
bestcost_validation = np.array(validation_Cost)
bestcost_test = np.array(test_Cost)

GA_PHIF = GA1.calculate_Obj_function(train_data, population, num_weights)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = GA1.calculate_fitness(GA_PHIF, Train_Tiab_PHIF,train)
print()

best_match_idx = np.where(fitness == np.min(fitness))
print("Best solution : ", population[best_match_idx, :])
print()
print("Best solution fitness : ", fitness[best_match_idx])


Train_GA_PHIF = GA1.GA_PHIF(train_data, population, num_weights, best_match_idx)
# print("Train_GA_PHIF",len(new_pop1))

Valida_GA_PHIF = GA1.GA_PHIF(validation_data,population, num_weights, best_match_idx)
# print("Validation_GA_PHIF",len(new_pop3))

Test_GA_PHIF = GA1.GA_PHIF(test_data,population, num_weights, best_match_idx)
# print("Test_GA_PHIF",len(new_pop2))

############################################################################################
############################################################################################

P_Error = GA1.prediction_error(Test_Tiab_PHIF,Test_GA_PHIF,test_data,test)
print("Total prediction error:", P_Error)

cost1 = GA1.best_cost(Train_Tiab_PHIF,Train_GA_PHIF,train)
print("Cost (Training data):", cost1)

cost2 = GA1.best_cost(Validation_Tiab_PHIF,Valida_GA_PHIF,validation)
print("Cost (Validation data):", cost2)

cost3 = GA1.best_cost(Test_Tiab_PHIF,Test_GA_PHIF,test)
print("Cost (Test data):", cost3)

print("Coeff of Det. Train",r2_score(Train_Tiab_PHIF, Train_GA_PHIF))
print("Coeff of Det. Test",r2_score(Test_Tiab_PHIF, Test_GA_PHIF))
print("Coeff of Det. Val",r2_score(Validation_Tiab_PHIF, Valida_GA_PHIF))

end = time.time()
print("Time in seconds:",end - start,"s")

############################################################################################
############################################################################################

# Output_plot1.Analysis_plot(Train_Tiab_PHIF, Test_Tiab_PHIF, Validation_Tiab_PHIF, Train_GA_PHIF, Test_GA_PHIF, Valida_GA_PHIF, TEPD, Depth, DEPT)
# plt.show()
# # Analysis_plot.fitness_plot(gen, bestfit)
# # plt.show()
# Output_plot1.cost_plot(gen, bestcost_train, bestcost_validation,bestcost_test)
# plt.show()
