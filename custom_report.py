from __future__ import division, print_function

import time

import numpy as np
import pandas as pd
import pickle

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """

    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""

    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class.
        Modified to save the printed information into a dictionary and then save it as a csv file"""

    def __init__(self, num_gens_to_store=1, file_name='neat-stats-'):
        self.dict = {
            'Generation': [],
            'Population_size': [],
            'Num_species': [],
            'list_of_[ID,age,size,fitness,adj_fit,stag]': [],
            'Total_extinctions': [],
            'Generation_time': [],
            'Average_time': [],
            'Average_fitness': [],
            'Stdev_fitness': [],
            'Best_genome_fitness': []}
#         self.show_species_detail = show_species_detail
        self.num_gens_to_store = num_gens_to_store
        self.file_name = file_name
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.dict['Generation'].append(generation)
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        self.dict['Num_species'][-1] = ns
#         if self.show_species_detail:
        print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
        sids = sorted(iterkeys(species_set.species))
        list_of_id_age_size_fitness_adjfit_stag = []
        print("   ID   age  size  fitness  adj fit  stag")
        print("  ====  ===  ====  =======  =======  ====")
        for sid in sids:
            s = species_set.species[sid]
            a = self.generation - s.created
            n = len(s.members)
            f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
            af = "--" if s.adjusted_fitness is None else "{:.3f}".format(
                s.adjusted_fitness)
            st = self.generation - s.last_improved
            print(
                "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(
                    sid, a, n, f, af, st))
            list_of_id_age_size_fitness_adjfit_stag.append(
                [sid, a, n, f, af, st])
        self.dict['list_of_[ID,age,size,fitness,adj_fit,stag]'][-1] = list_of_id_age_size_fitness_adjfit_stag
#         else:
#             print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        self.dict['Total_extinctions'][-1] = self.num_extinctions
        self.dict['Generation_time'][-1] = elapsed
        self.dict['Average_time'][-1] = average
        if len(self.generation_times) > 1:
            print(
                "Generation time: {0:.3f} sec ({1:.3f} average)".format(
                    elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))
        ngen = self.dict['Generation'][-1]
        if ngen % self.num_gens_to_store == 0:
            self.dump(self.file_name + str(ngen))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        self.dict['Population_size'].append(len(population))
        self.dict['Num_species'].append('--')
        self.dict['list_of_[ID,age,size,fitness,adj_fit,stag]'].append([])
        self.dict['Total_extinctions'].append('--')
        self.dict['Generation_time'].append('--')
        self.dict['Average_time'].append('--')
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print(
            'Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(
                fit_mean,
                fit_std))
        self.dict['Average_fitness'].append(fit_mean)
        self.dict['Stdev_fitness'].append(fit_std)
        print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(
            best_genome.fitness, best_genome.size(), best_species_id, best_genome.key))
        self.dict['Best_genome_fitness'].append(best_genome.fitness)

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print(
            '\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
                self.generation,
                best.size()))

    def species_stagnant(self, sid, species):
        #         if self.show_species_detail:
        print(
            "\nSpecies {0} with {1} members is stagnated: removing it".format(
                sid, len(
                    species.members)))

    def info(self, msg):
        print(msg)

    def save_table(self, file_name):
        ngen = self.dict['Generation'][-1]
        self.dump(self.file_name + str(ngen))
        df = pd.DataFrame(self.dict)
        df.to_csv(file_name + '.csv', header=True)

    def dump(self, file_name):
        print('saving stats with name:' + file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_outside(self, file_name):
        with open(file_name, 'rb') as f:
            tmp = pickle.load(f)
            self.__dict__.update(tmp)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            tmp = pickle.load(f)
            self.__dict__.update(tmp)
            self.dict['Generation'] = self.dict['Generation'][:-1]
            self.dict['Population_size'] = self.dict['Population_size'][:-1]
            self.dict['Num_species'] = self.dict['Num_species'][:-1]
            self.dict['list_of_[ID,age,size,fitness,adj_fit,stag]'] = self.dict['list_of_[ID,age,size,fitness,adj_fit,stag]'][:-1]
            self.dict['Total_extinctions'] = self.dict['Total_extinctions'][:-1]
            self.dict['Generation_time'] = self.dict['Generation_time'][:-1]
            self.dict['Average_time'] = self.dict['Average_time'][:-1]
            self.dict['Average_fitness'] = self.dict['Average_fitness'][:-1]
            self.dict['Stdev_fitness'] = self.dict['Stdev_fitness'][:-1]
            self.dict['Best_genome_fitness'] = self.dict['Best_genome_fitness'][:-1]
