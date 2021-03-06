from __future__ import print_function

import multiprocessing
import os

import neat
import numpy as np
import gym

from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import custom_report
# import custom_checkpoint

env = gym.make('CarRacing-v0')

NUM_GENERATIONS = 101
PATIENCE = 0


def map_image(p):
    if p[1] <= 107:
        return 1
    return 0


def preprocess(frame):
    """
    This function receives a frame of the game and
        - converts all pixes to {0, 1} according to whther they are rouad or not.
        - performs mean pooling in kernels of 10x10, to reduce dimensionality.
        - flattens the result into an array 1D of size 90
    """
    # Simplify the map by setting the road to 1 and everything else to 0
    frame = np.array([[map_image(pixel) for pixel in row] for row in frame])
    frame = frame[:83, :]
    frame = block_reduce(frame, (10, 10), np.mean)
    frame = frame.flatten()
    return frame


def eval_network(net, frame):
    """ This function feeds the net with an already preprocessed frame """
    # checks if the frame has been already preprocessed
    assert (frame.shape == (90,))
    result = net.activate(frame)
    # checks that the output consists of three numbers
    assert (len(result) == 3)
    return result


def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    returns one float: the genome's fitness.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 6.0
    frame = env.reset()
    frame = preprocess(frame)
    last_frame = frame
    action = eval_network(net, frame)
    done = False
    steps = 0
    patience_count = 0
    while not done:
        env.render()
        frame, reward, done, _ = env.step(action)
        total_reward += reward
        frame = preprocess(frame)
        # If the road disappears of the map, exit the game with reward -= 80
        if not frame.any():
            done = True
            total_reward -= 80
        # If the car stops exit the game with the reward that we would have if
        # we have wait to the end.
        if np.array_equal(last_frame, frame):
            if patience_count == PATIENCE:
                done = True
                total_reward += -0.1 * (1000 - steps)
            else:
                patience_count += 1
        else:
            patience_count = 0
        action = eval_network(net, frame)
        last_frame = frame
        steps += 1

    return total_reward


def run(config_file):
    """ Main function that runs the neat platform
        - Loads the config file
        - Initializes the population according to the parameters of the config file
        - Add a Reporter to collect statistics of the runs
        - Add a Checkpointer to save (using picke):
            generation, config, population, species_set, rndstate
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    custom_stats = custom_report.StdOutReporter()
    p.add_reporter(custom_stats)

    # save a checkpointer each generation
    p.add_reporter(neat.Checkpointer(1))

    # Run using parallelizing with thenumber of processors for NUM_GENERATIONS
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, NUM_GENERATIONS)
    # save the results of the Statistics collected by the Reporter into a csv
    custom_stats.save_table('stats_table')


if __name__ == '__main__':
    """Determine the path to the configuration file """
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'baseline_config_file')
    run(config_path)
