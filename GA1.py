# program code is edited based on the follwing resources
# https://pythonhealthcare.org/2018/10/01/94-genetic-algorithms-a-simple-genetic-algorithm/
# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

import numpy as np
import random

np.set_printoptions(precision=6)

# The function to create initial population
def create_starting_population(solutions, weights):
    # Set up an initial array of all zeros
    pop_size = (solutions, weights)
    population= np.random.normal(size=pop_size)
    return population
# print(create_starting_population(5, 3))



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

############# The general Genetic Algorithm Functions ########################

# Defining the objective function
def calculate_Obj_function(train_data, population, num_weights):

    GA_PHIF_arr = []

    for data in range(population.shape[0]):
        # GA_PHIF = np.empty([train_data.shape[0], train_data.shape[0]])
        pop = np.empty([num_weights])
        pop = population[data,:]
        #print(pop.shape,train_data.shape, population[data,:].shape)
        for dt in range(train_data.shape[0]):
            val = np.sum(train_data[dt,:]*pop)
            GA_PHIF_arr.append(np.tanh(val))
        GA = np.array(GA_PHIF_arr)
        # print(GA)
    GA_PHIF = GA.reshape(population.shape[0], train_data.shape[0])
    return GA_PHIF # should be positive Error

# Defining fitness function
def calculate_fitness(GA_PHIF, Train_Tiab_PHIF,train):

    score = []
    for i in range(Train_Tiab_PHIF.shape[0]):
        # score.append(np.sqrt(np.mean((GA_PHIF[i, :] - Train_Tiab_PHIF[i]) ** 2)))
        score.append(np.sqrt(np.sum((GA_PHIF[:, i] - Train_Tiab_PHIF[i]) ** 2) / train))
    fitness = np.array(score)
    return fitness

# Defining the Selection operator
# to Select the best individuals in the current generation as parents (half of the total population)
def select_mating_pool(population, fitness, num_parents):

    parents = np.empty((num_parents, population.shape[1]))  #c.shape[1] Gives number of columns

    for parent_no in range(num_parents):

        min_score_idx = np.where(fitness == np.min(fitness)) ### identify the index where the fitness value is minimum
        min_score_idx = min_score_idx[0][0]   # select and use the index
        parents[parent_no, :] = population[min_score_idx, :]

        fitness[min_score_idx] = 9999999  #Erasing fitness value where the fitness value is minimum

    return parents

# Defining Crossover operator
# to Produce children from parents – crossover
def breed_by_crossover(parents, Train_Tiab_PHIF, train_data, offspring_size):
    offspring = np.empty(offspring_size)  ## edited

    a = 0.7  #uint8	Unsigned integer (0 to 255)

    for child_ID_no in range(offspring_size[0]):
        # offspring1, offspring2 = np.empty(int((parents.shape[0])/2),(offspring_size.shape[1]))

        parent1_idx = child_ID_no % parents.shape[0]        # Index of the first parent to mate.
        parent2_idx = (child_ID_no + 1) % parents.shape[0]  # Index of the second parent to mate.

        if child_ID_no % 1 == 0:
            offspring[parent1_idx, :] = a*parents[parent1_idx, :] + (1-a)*parents[parent2_idx, :]
        else:
            offspring[parent1_idx, :] = (1-a)*parents[parent1_idx, :] + a*parents[parent2_idx, :]
        # pop = offspring[parent1_idx, :]
        # GA_PHIF_arr = []
        # for dt in range(train_data.shape[0]):
        #     val = np.sum(train_data[dt,:]*pop)
        #     GA_PHIF_arr.append(val)
        # GA = np.array(GA_PHIF_arr)
        # Train_Tiab_PHIF = Train_Tiab_PHIF.reshape((GA.shape))
        # GA = (GA - Train_Tiab_PHIF[0])
        # grad = []
        # for i in range(GA.shape[0]):
        #     grad.append(GA[i] * train_data[i])
        # #grad = GA * train_data
        # grad = np.array(grad)
        # grad = np.sum(grad, axis=0)
        # offspring[parent1_idx, :] = offspring[parent1_idx, :] - 0.01*grad
    return offspring

# Defining Mutation operator
# Mutation changes a single gene in each offspring randomly. (Random resetting)
def randomly_mutate_population(offspring_mutation):

    # Apply random mutation
    for idx in range(offspring_mutation.shape[0]):
        chromosome_length = len(offspring_mutation[0,:])
        for t_ in range(3):
            mutation_point = random.randint(1, chromosome_length-1)
            random_value = np.random.normal(size=1)                      # The random value to be added to the gene.
            offspring_mutation[idx, mutation_point] = offspring_mutation[idx, mutation_point] + random_value ##error
    # Return mutated offspring
    return offspring_mutation


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#Calculate GA fracture porosity using train, validation and test data
def GA_PHIF(split_data, population, num_weights, best_match_idx):
    PHIF_arr = []
    for n in range(split_data.shape[0]):
        new_pop1 = np.empty([split_data.shape[0], num_weights])
        new_pop1[0:,:] = population[best_match_idx, :]
        PHIF_arr.append(np.sum(split_data[n,:]*new_pop1[n,:]))
    GA_Frac_Porosity = np.array(PHIF_arr)
    return GA_Frac_Porosity


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# Defining Total prediction error function
def prediction_error(Test_Tiab_PHIF,Test_GA_PHIF,test_data,test):
    MSE_sum = 0
    for r in range(test_data.shape[0]):
        MSE_sum += (Test_Tiab_PHIF[r] - Test_GA_PHIF[r])** 2
    P_Error = MSE_sum/test
    return P_Error

# Defining Cost function
def best_cost(Train_Tiab_PHIF,Train_GA_PHIF,train):
    cost_sum = 0
    for r in range(Train_GA_PHIF.shape[0]):
        cost_sum += (Train_Tiab_PHIF[r] - Train_GA_PHIF[r])**2
    cost = cost_sum/(2*train)
    return cost

# Defining Cost function to esimate the cost at each generation for train, validation and test data
def cost(Tiab_PHIF,population, best_match_idx,total_numbers_data, splited_data, num_weights):
    PHIF_arr = []
    for n in range(splited_data.shape[0]):
        new_pop1 = np.empty([splited_data.shape[0], num_weights])
        new_pop1[0:, :] = population[best_match_idx, :]
        PHIF_arr.append(np.sum(splited_data[n, :] * new_pop1[n, :]))
    GA_PHIF = np.array(PHIF_arr)
    cost_sum = 0
    for r in range(GA_PHIF.shape[0]):
        cost_sum += (Tiab_PHIF[r] - GA_PHIF[r])**2
    cost_plot = cost_sum/(2*total_numbers_data)
    return cost_plot







