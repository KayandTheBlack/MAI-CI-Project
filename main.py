#
import gym
import neat
import numpy as np
from neat.parallel import ParallelEvaluator

from functools import partial
from skimage.measure import block_reduce


env = gym.make("CarRacing-v0")
#print(env.action_space) returns expected 
conf_file = "example_config_file"
MAX_GENERATIONS  = 10
NUM_WORKERS = 16 # Parallelise evaluations

# Sets config to the one in the config file, rest to defaults
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
             neat.DefaultSpeciesSet, neat.DefaultStagnation,
             conf_file)

def preprocess(input):
    # Performs mean pooling in kernels of 2x2x3, to reduce dimmensionality. Resulting shape then flattened and fed into net
    # Note that the config file MUST have num_inputs = dim(flattened)
    input = block_reduce(input, (2,2,3), np.mean)
    input = input.flatten()
    #print(input.shape)
    return input


def eval_network(net, input):
    input = preprocess(input)
    assert (input.shape == (2304,)) # Debugger for ensuring net receives appropiate input postpreprocess
    result = net.activate(input)
    assert (len(result) == 3) # Debugger for ensuring output are 3 actions (continuous, as env.action_space is Box 3)
    return result

def eval_single_genome(genome, genome_config):
    # Creates new NN with the genome and the config
    global env
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0
    observation = env.reset() # Start new game
    action = eval_network(net, observation) # Obtain the next action
    done = False
    while not done:
        env.render()
        observation, reward, done, _ = env.step(action) 
        # Perform action in current environ
        # Returns next observation, the current reward, whether we are done and some other info we discard.
        total_reward += reward
        action = eval_network(net, observation)
    return total_reward

def fitness(genomes, neat_config):
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)
    parallel_evaluator.evaluate(genomes, neat_config)

def run_neat(env, config):
    global MAX_GENERATIONS
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False)) # Write to terminal

    winner = p.run(fitness, n=MAX_GENERATIONS)

run_neat(env, config)
