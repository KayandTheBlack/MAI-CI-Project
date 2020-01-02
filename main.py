#
import gym
import neat
import numpy as np
from neat.parallel import ParallelEvaluator

from functools import partial


env = gym.make("CarRacing-v0")
conf_file = "example_config_file"
MAX_GENERATIONS  = 10
NUM_WORKERS = 16 # Parallelise evaluations

# Sets config to the one in the config file, rest to defaults
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
             neat.DefaultSpeciesSet, neat.DefaultStagnation,
             conf_file)

def eval_single_genome(genome, genome_config)

def eval_many_genomes(genomes, neat_config):
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)
    parallel_evaluator.evaluate(genomes, neat_config)



def run_neat(env, config):
    global MAX_GENERATIONS
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False)) # Write to terminal

    final_fitness = partial(eval_many_genomes, eval_single_genome)
    # Merges the parallelised many genomes and the single_genome into one, as req. by run.
    winner = p.run(, n=MAX_GENERATIONS)

run_neat(env, config)
