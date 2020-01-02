#
import argparse
import datetime
import gym
import neat
import numpy as np
from neat.parallel import ParallelEvaluator

from functools import partial
from skimage.measure import block_reduce


CONFIG_FILENAME = "example_config_file"
MAX_GENERATIONS  = 5
NUM_WORKERS = 16 # Parallelise evaluations
CHECKPOINT = None
CHECKPOINT_GENERATION_INTERVAL = 1
CHECKPOINT_PREFIX = None
VISUALIZE = False
VERBOSE = 0
RENDER = True


def preprocess(input):
    # Performs mean pooling in kernels of 2x2x3, to reduce dimmensionality. Resulting shape then flattened and fed into net
    # Note that the config file MUST have num_inputs = dim(flattened)
    input = block_reduce(input, (2,2,3), np.mean)
    input = input.flatten()
    # print(input.shape)
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
        if not RENDER:
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
    if CHECKPOINT is not None:
        print("Resuming from checkpoint: {}".format(CHECKPOINT))
        p = neat.Checkpointer.restore_checkpoint(CHECKPOINT)
    else:
        print("Starting from scratch")
        p = neat.Population(config)

    stats = neat.StatisticsReporter()
    
    p.add_reporter(neat.StdOutReporter(False)) # Write to terminal
    p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))


    winner = p.run(fitness, n=MAX_GENERATIONS)
    generate_stat_plots(stats, winner)


def generate_stat_plots(stats, winner):
    if VISUALIZE:
        pass


def parse_args():
    global CHECKPOINT
    global CHECKPOINT_GENERATION_INTERVAL
    global CHECKPOINT_PREFIX
    global MAX_GENERATIONS
    global NUM_WORKERS
    global CONFIG_FILENAME
    global VISUALIZE
    global VERBOSE
    global RENDER

    parser = argparse.ArgumentParser()    
    parser.add_argument('--checkpoint', nargs='?', default=CHECKPOINT, help='The filename to restart from')
    parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX, help='Prefix for the filename')
    parser.add_argument('--checkpoint-gi', nargs='?', default=CHECKPOINT_GENERATION_INTERVAL, help='generation between saves')
    parser.add_argument('--max-generations', nargs='?', default=MAX_GENERATIONS, help='Maximum number of generations')
    parser.add_argument('--num-workers', nargs='?', default=NUM_WORKERS, help='Parallelise evaluations')
    parser.add_argument('--config', nargs='?', default=CONFIG_FILENAME, help='The name of the configuration file of neat')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='True for visualization of the net')
    parser.add_argument('--render', dest='render', default=True, action='store_false', help='True for rendering')
    parser.add_argument('--verbose', nargs='?', default=VERBOSE, help='0 - non verbose, 1 verbose')


    command_line_args = parser.parse_args()

    CHECKPOINT = command_line_args.checkpoint
    CHECKPOINT_GENERATION_INTERVAL = command_line_args.checkpoint_gi
    CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix
    MAX_GENERATIONS = command_line_args.max_generations
    NUM_WORKERS = command_line_args.num_workers
    CONFIG_FILENAME = command_line_args.config
    VISUALIZE = command_line_args.visualize
    VERBOSE = command_line_args.verbose
    RENDER = command_line_args.render

    
    return command_line_args


if __name__ == '__main__':

    parse_args()
    # Sets config to the one in the config file, rest to defaults
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
             neat.DefaultSpeciesSet, neat.DefaultStagnation,
             CONFIG_FILENAME)
    env = gym.make("CarRacing-v0")

    if CHECKPOINT_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        CHECKPOINT_PREFIX = "cp_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_gen_"

    run_neat(env, config)

