import deap
from deap import tools
from deap import base, creator
import time
import numpy as np
import pandas as pd
import random
import copy
from random import randrange
import pyswarms as ps
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import math
from pyswarms.backend.topology import Star

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class Evolution:

    @staticmethod
    def create_population(population_size, ind_size, fixed_p=None):
        pop = []
        for i in range(population_size):
            if (not fixed_p):
                zero_p = random.uniform(0, 1)
            else:
                zero_p = random.uniform(fixed_p[0], fixed_p[1])
            pop.append(deap.creator.Individual(np.random.choice([0, 1], size=(ind_size,), p=[zero_p, (1 - zero_p)])))
        return list(pop)

    @staticmethod
    def create_toolbox(task, target_dataset, baseline_individual, baseline_full_data=None, original_fitness_time=None):
        toolbox = base.Toolbox()
        toolbox.register("mate", Evolution.HUX)
        toolbox.register("select", tools.selTournament, tournsize=1)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1/3)

        if (task == 'feature_selection'):
            toolbox.register("evaluate", Evolution.evaluate, task=task, target_dataset=target_dataset,
                             baseline_individual=baseline_individual)

        if (task == 'instance_selection'):
            toolbox.register("evaluate", Evolution.evaluate_selected_instances, baseline_full_data=baseline_full_data,
                             baseline_individual=baseline_individual, GA_representation=True, original_fitness_time=original_fitness_time)
        return toolbox

    @staticmethod
    def hammingDistance(ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (len(ind1) - (np.sum(np.equal(ind1, ind2))))

    @staticmethod
    def HUX(ind1, ind2, fixed=True):
        # index variable
        idx = 0

        # Result list
        res = []

        # With iteration
        for i in ind1:
            if i != ind2[idx]:
                res.append(idx)
            idx = idx + 1
        if (len(res) > 1):
            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            oldInd1 = copy.copy(ind1)

            for i in indx:
                ind1[i] = ind2[i]

            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            for i in indx:
                ind2[i] = oldInd1[i]
        return ind1, ind2

    @staticmethod
    def f(x, task, target_dataset, baseline_individual):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        #print(x)
        #print()
        x = np.where(x > 0.5, 1, 0)

        #if np.count_nonzero(x[0]) == 0:
        #    print('test', x[0])
        #else:
        #    print('test1', x[0])

        n_particles = x.shape[0]

        j = [Evolution.evaluate_PSO(x[i], task, target_dataset, baseline_individual) for i in range(n_particles)]
        return np.array(j)

    @staticmethod
    def evaluate_PSO(individual, task, target_dataset, baseline_individual):
        return -1 * Evolution.evaluate(individual, task, target_dataset, baseline_individual)[0]

    @staticmethod
    def select_instances(n_individual, baseline_individual, minimum_sample_size=5000):
        start = time.time()
        baseline_full_data = []
        original_fitness_time = 0
        start = time.time()
        for i in range(n_individual):
            tmp = copy.copy(baseline_individual)
            tmp.divide_dataset(baseline_individual.clf,
                               normalize=True,
                               shuffle=False,
                               all_features=False,
                               all_instances=True,
                               evaluate=True,
                               partial_sample=False)
            # tmp.setInstances(select_k_instances(int(oss), len(tmp.X_train)))
            # print(tmp.features)
            baseline_full_data.append(tmp)
        end = time.time()
        original_fitness_time = end - start
        usefulness_df = pd.DataFrame(columns=['sample_size', 'correlation'])

        sample_size = baseline_individual.X_train.shape[0] // 2
        start = time.time()
        approximate_population = []
        best = np.inf
        improving = True
        while improving and sample_size > minimum_sample_size:
            improving = False
            c1 = copy.copy(baseline_individual)
            c1.divide_dataset(baseline_individual.clf,
                              normalize=True,
                              shuffle=False,
                              all_features=True,
                              all_instances=True,
                              evaluate=False,
                              partial_sample=sample_size)
            # tmp.setInstances(select_k_instances(int(oss), len(tmp.X_train)))
            start = time.time()
            approximate_performance = []
            approximation_time = 0
            start = time.time()
            for c in baseline_full_data:
                c1.set_features(c.features)
                c1.fit_classifier()
                c1.set_validation_accuracy()
                # print(c1.get_validation_accuracy())
                approximate_population.append(c1)

                approximate_performance.append(c1.get_validation_accuracy())
            end = time.time()
            approximation_time = end - start
            # print(sample_size, end-start)
            full_data_performance = []
            for c in baseline_full_data:
                full_data_performance.append(c.get_validation_accuracy())

            coef, p = spearmanr(full_data_performance, approximate_performance)
            r = coef
            if (r < 0):
                r = 0
            num = n_individual
            stderr = 1.0 / math.sqrt(num - 3)
            delta = 1.96 * stderr
            lower = math.tanh(math.atanh(r) - delta)
            # print(coef, lower)
            # distance = (1-coef)+sample_size/baseline_individual.X_train.shape[0]
            distance = (1 - lower) + approximation_time / original_fitness_time
            print(sample_size, coef, distance)
            # print((c1.instances))
            if (distance < best):
                best = distance
                selected_instances = c1.instances
                best_sample_size = sample_size
                improving = True
            row = [sample_size, coef]
            usefulness_df.loc[len(usefulness_df)] = row
            sample_size = sample_size // 2

        return selected_instances, baseline_full_data

    @staticmethod
    def evaluate_selected_instances(individual, baseline_full_data, baseline_individual, GA_representation, original_fitness_time, plot=False):
        instance_index = individual
        if (GA_representation):
            selected = np.array(individual)
            instance_index = list(np.where(selected == 1)[0])
        # c = copy.copy(baseline_individual)
        approximate_population = []
        start = time.time()
        for c in baseline_full_data:
            c1 = copy.copy(baseline_individual)
            c1.set_features(c.features)
            c1.set_instances(instance_index)
            c1.fit_classifier()
            c1.set_validation_accuracy()
            approximate_population.append(c1)
        end =time.time()
        approximation_time = end -start
        approximate_performance = []
        for c in approximate_population:
            approximate_performance.append(c.get_validation_accuracy())
        full_data_performance = []
        for c in baseline_full_data:
            full_data_performance.append(c.get_validation_accuracy())
        coef, p = spearmanr(full_data_performance, approximate_performance)
        r = coef
        num = len(baseline_full_data)
        stderr = 1.0 / math.sqrt(num - 3)
        delta = 1.96 * stderr
        #lower = math.tanh(math.atanh(r) - delta)
        #print(len(instance_index), lower)
        distance = (1 - coef) + approximation_time / original_fitness_time
        if (plot):
            plt.scatter(approximate_performance, full_data_performance)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylabel('original function')
            plt.xlabel('approximation')
            plt.show()
            # plt.scatter(range(len(approximate_performance)), approximate_performance)
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.ylabel('fitness')
            index, value = max(enumerate(full_data_performance), key=operator.itemgetter(1))
            plt.scatter(index, value, marker='P', s=100, color='b', label='optimum')
            index, value = max(enumerate(approximate_performance), key=operator.itemgetter(1))
            plt.scatter(index, value, marker='*', s=100, color='b', label='approximation optimum')
            plt.plot(range(len(full_data_performance)), full_data_performance, 'b', label='original function')
            plt.plot(range(len(approximate_performance)), approximate_performance, '--', label='approximation')
            plt.scatter(range(len(full_data_performance)), full_data_performance, s=20)
            plt.ylim((0.15, 0.32))
            # plt.yticks(fontsize=8)
            plt.legend(loc="lower left", prop={'size': 8})
            plt.show()
        return -1 * distance,

    @staticmethod
    def evaluate(individual, task, target_dataset, baseline_individual):
        selected = np.array(individual)
        selected_indexes = list(np.where(selected == 1)[0])
        c = copy.copy(baseline_individual)
        if (len(selected_indexes) > 0):
            if (task == 'instance_selection'):
                c.set_instances(selected_indexes)
            if (task == 'feature_selection'):
                c.set_features(selected_indexes)
            if (task == 'baseline'):
                c.set_instances(range(0, c.X_train.shape[0]))
                c.set_features(range(0, c.X_train.shape[1]))
            c.fit_classifier()
            if (target_dataset == 'validation'):
                c.set_validation_accuracy()
                return c.get_validation_accuracy(),
            if (target_dataset == 'test'):
                c.set_test_accuracy()
                return c.get_test_accuracy(),
            if (target_dataset == 'cv'):
                c.set_CV()
                return c.get_CV(),
        else:
            return 0,

    @staticmethod
    def CHCqx(baseline_individual, f, n_individual, f_no_change, population_size=50, verbose=1):
        gaqx_log_df = pd.DataFrame(columns=['ind', 'time', 'fitness'])

        start = time.time()
        selected_instances, baseline_full_data = Evolution.select_instances(n_individual, baseline_individual)
        if (not population_size):
            population_size = baseline_individual.X_train.shape[1]

        counter = 0
        gaqx_individual = copy.deepcopy(baseline_individual)

        gaqx_individual.set_instances(selected_instances)

        task = 'feature_selection'
        target_dataset = 'validation'
        ind_size = gaqx_individual.X_train.shape[1]
        toolbox = Evolution.create_toolbox(task, target_dataset, gaqx_individual, baseline_full_data)
        population = Evolution.create_population(population_size, ind_size)

        evaluated_population = []
        best = 0
        no_change = 0
        d = ind_size // 4
        while (no_change < f_no_change):
            no_change += 1
            log_df, population, d = Evolution.CHC(gaqx_individual, toolbox, d, population, max_generations=f, verbose=1)
            fitness_sum = 0
            ind_id = 0
            for ind in population:
                ind_id += 1
                if ind not in evaluated_population:
                    evaluated_population.append(ind)
                    fitness = Evolution.evaluate(ind, task, target_dataset, baseline_individual)[0]
                    fitness_sum = fitness_sum + fitness
                    if fitness > best:
                        gaqx_time = time.time() - start
                        row = [ind, gaqx_time, fitness]
                        gaqx_log_df.loc[len(gaqx_log_df)] = row
                        if (verbose):
                            print(ind_id, row)
                        best = fitness
                        no_change = 0

        return gaqx_log_df, baseline_full_data

    @staticmethod
    def CHC(dataset, toolbox, d, population=False, population_size=40,
            divergence=3,
            max_generations=np.inf, max_no_change=np.inf, timeout=np.inf, optimization_task = 'feature_selection',
            stop=np.inf, verbose=0):
        start = time.time()
        end = time.time()

        if (optimization_task == 'feature_selection'):
            ind_size = len(dataset.features)
        elif (optimization_task == 'instance_selection'):
            ind_size = len(dataset.instances)

        #toolbox = Evolution.create_toolbox(optimization_task, evaluation, dataset)

        generationCounter = 0
        evaulationCounter = 0
        best = -1 * np.inf
        noChange = 0

        # if (not d):
        #    d = d0
        logDF = pd.DataFrame(
            columns=(
                'generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))


        if (not population):

                population = Evolution.create_population(population_size, ind_size)
        population_size = len(population)

            # for ind in population:
            #    print(ind)
            #    print(Representation.Maxout(np.array(ind)).s2(1))

            # calculate fitness tuple for each individual in the population:
            # fitnessValues = list(map(toolbox.evaluate, population))
        evaluatedIndividuals = [ind for ind in population if ind.fitness.valid]
        bestInd = toolbox.clone(population[0])
        updated = False
        for individual in evaluatedIndividuals:
            if (best < individual.fitness.values[0]):
                noChange = 0
                best = individual.fitness.values[0]
                bestInd = toolbox.clone(individual)
                bestTime = time.time()
                updated = True

            # print(time.time()-start)
        if (time.time() - start) > timeout:
            if (updated):
                print('log1', bestInd)
                row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                       bestInd, d]
                logDF.loc[len(logDF)] = row
                updated = False
            return logDF, population, d

        freshIndividuals = [ind for ind in population if not ind.fitness.valid]
        for individual in freshIndividuals:
            # print(earlyTermination)
            individual.fitness.values = toolbox.evaluate(individual)
            evaulationCounter += 1
            if (best < individual.fitness.values[0]):
                noChange = 0
                best = individual.fitness.values[0]
                bestTime = time.time()
                bestInd = toolbox.clone(individual)
                updated = True
                #print(best)
            # print(time.time()-start)
            if updated:
                    row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                           bestInd, d]
                    logDF.loc[len(logDF)] = row
                    updated = False
            if (time.time() - start) > timeout:
                return logDF, population, d


        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]
        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        #d = len(population[0]) // 4
        d0 = len(population[0]) // 4
        #d0 = copy.deepcopy(d)
        populationHistory = []
        for ind in population:
            populationHistory.append(ind)

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < max_generations and noChange < max_no_change and (
                end - start) < timeout:
            # update counter:
            generationCounter = generationCounter + 1

            #for ind in population:
            #    print(ind, ind.fitness.values)
            #print()

            # apply the selection operator, to select the next generation's individuals:
            # offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, population))
            random.shuffle(offspring)

            newOffspring = []

            newOffspringCounter = 0

            # apply the crossover operator to pairs of offspring:
            numberOfPaired = 0
            numberOfMutation = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if Evolution.hammingDistance(child1, child2) > d and d > 0:
                    # print('Before')
                    # print(child1)
                    # print(child2)
                    toolbox.mate(child1, child2)
                    numberOfPaired += 1
                    newOffspringCounter += 2
                    addChild = True
                    for ind in populationHistory:
                        if (Evolution.hammingDistance(ind, child1) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child1)
                        newOffspring.append(child1)
                    addChild = True
                    for ind in populationHistory:
                        if (Evolution.hammingDistance(ind, child2) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child2)
                        newOffspring.append(child2)
                    # print('history length', len(populationHistory))
                    # print('After')
                    # print(child1)
                    # print(child2)
                    # print()
                    del child1.fitness.values
                    del child2.fitness.values
            # print('this is d', d)
            if (d == 0):
                #print('shit', d0, d)
                d = copy.deepcopy(d0)
                #print('shit', d0, d)
                newOffspring = []
                bestIndividual = tools.selBest(population, 1)[0]
                while (numberOfMutation < len(population)):
                    mutant = toolbox.clone(bestIndividual)
                    numberOfMutation += 1
                    toolbox.mutate(mutant)
                    populationHistory.append(mutant)
                    newOffspring.append(mutant)
                    del mutant.fitness.values

            # if (newOffspringCounter == 0 and d > 0):
            #    d -= 1
            noChange += 1
            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
            # freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual in freshIndividuals:
                individual.fitness.values = toolbox.evaluate(individual)
                evaulationCounter += 1
                if (best < individual.fitness.values[0]):
                    noChange = 0
                    best = individual.fitness.values[0]
                    bestTime = time.time()
                    bestInd = toolbox.clone(individual)
                    updated = True
                # print(time.time()-start)
                if (time.time() - start) > timeout:
                    if (updated):
                        row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                               bestInd, d]
                        logDF.loc[len(logDF)] = row
                    # row = [generationCounter, (end - start), np.round(best, 4), -1, evaulationCounter,
                    #       individual, d]
                    # logDF.loc[len(logDF)] = row
                    return logDF, population, d

            # evaulationCounter = evaulationCounter + len(freshIndividuals)

            if (numberOfMutation == 0):
                oldPopulation = copy.copy(population)
                population[:] = tools.selBest(population + newOffspring, population_size)
                differentPopulation = False
                for index in range(0, len(population)):
                    if (Evolution.hammingDistance(oldPopulation[index], population[index]) != 0):
                        differentPopulation = True
                #print(differentPopulation)
                if (not differentPopulation):
                    d -= 1
            else:
                bestIndividual = tools.selBest(population, 1)
                population[:] = tools.selBest(bestIndividual + newOffspring, population_size)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            # if (best >= maxFitness):
            #    noChange += 1
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            #print(np.round(maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:',
            #      numberOfMutation, ' d:', d, ' no change:', noChange)
            # print('new', newOffspringCounter)
            #print()
            end = time.time()
            if (updated):
                row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                       bestInd, d]
                logDF.loc[len(logDF)] = row
                updated = False
        end = time.time()
        return logDF, population, d

    @staticmethod
    def PSOqx(baseline_individual, options, f, n_individual, f_no_change, n_particles=None, verbose=1):
        gaqx_log_df = pd.DataFrame(columns=['ind', 'time', 'fitness'])

        start = time.time()
        selected_instances, baseline_full_data = Evolution.select_instances(n_individual, baseline_individual)

        gaqx_individual = copy.deepcopy(baseline_individual)
        gaqx_individual.set_instances(selected_instances)
        ind_size = gaqx_individual.X_train.shape[1]
        if (not n_particles):
            n_particles = ind_size

        task = 'feature_selection'
        target_dataset = 'validation'
        population = Evolution.create_population(n_particles, ind_size)
        population = np.array(population)
        population = population.astype(float)
        opt = ps.single.GeneralOptimizerPSO(n_particles=n_particles, dimensions=ind_size, init_pos=population,
                                            topology=Star(), options=options)


        evaluated_population = []
        best = 0
        no_change = 0
        while (no_change < f_no_change):
            no_change += 1
            opt.optimize(Evolution.f, iters=f, verbose=0, task=task, target_dataset=target_dataset,
                                     baseline_individual=gaqx_individual)
            fitness_sum = 0
            ind_id = 0
            for ind in opt.swarm.pbest_pos[np.argsort(opt.swarm.pbest_cost)]:
                ind = np.where(ind > 0.5, 1, 0)
                ind = list(ind)
                ind_id += 1
                if ind not in evaluated_population:
                    evaluated_population.append(ind)
                    fitness = Evolution.evaluate(ind, task, target_dataset, baseline_individual)[0]
                    fitness_sum = fitness_sum + fitness
                    if fitness > best:
                        gaqx_time = time.time() - start
                        row = [ind, gaqx_time, fitness]
                        gaqx_log_df.loc[len(gaqx_log_df)] = row
                        if (verbose):
                            print(ind_id, row)
                        best = fitness
                        no_change = 0

        return gaqx_log_df, baseline_full_data

    @staticmethod
    def PSO(baseline_individual, options, n_particles, timeout=np.inf, steps=np.inf, steps_no_change=10, verbose=1):
        gaqx_log_df = pd.DataFrame(columns=['ind', 'time', 'fitness'])
        ind_size = baseline_individual.X_train.shape[1]
        task = 'feature_selection'
        target_dataset = 'validation'
        population = Evolution.create_population(n_particles, ind_size)
        population = np.array(population)
        population = population.astype(float)
        opt = ps.single.GeneralOptimizerPSO(n_particles=n_particles, dimensions=ind_size, init_pos=population,
                                            topology=Star(), options=options)
        step = 0
        best_cost = np.inf
        no_change = 0
        start = time.time()
        while (time.time() - start < timeout and step < steps and no_change < steps_no_change):
            no_change += 1
            cost, pos = opt.optimize(Evolution.f, iters=1, verbose=0, task=task, target_dataset=target_dataset,
                                     baseline_individual=baseline_individual)
            step += 1
            ind = np.where(pos > 0.5, 1, 0)
            if (cost < best_cost):
                end = time.time()
                row = [ind, end-start, cost]
                gaqx_log_df.loc[len(gaqx_log_df)] = row
                no_change = 0
                best_pos = pos
                best_cost = cost
        return gaqx_log_df