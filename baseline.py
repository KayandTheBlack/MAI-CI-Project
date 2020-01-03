from __future__ import print_function

import multiprocessing
import os

import visualize

import neat
import numpy as np
import gym

from skimage.measure import block_reduce
import matplotlib.pyplot as plt

env = gym.make('CarRacing-v0')

def preprocess(frame):
    """
    This function receives a frame of the game and performs mean pooling in kernels od 2x2x3, to reduce dimensionality. Resulting 
    """
    frame = frame[:83, :]
    plt.imshow(frame)
    plt.show()
    frame = block_reduce(frame, (2, 2, 3), np.mean)
    frame = frame.flatten()
#     print(frame.shape)
    return frame


def eval_network(net, frame):
    frame = preprocess(frame)
    assert (frame.shape == (2016,))
    result = net.activate(frame)
    assert (len(result) == 3)
    return result

def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0.0
    frame = env.reset()
    action = eval_network(net, frame)
    done = False
    while not done:
        env.render()
        frame, reward, done, _ = env.step(action)
        total_reward += reward
        action = eval_network(net, frame)
    return total_reward

def run(config_file):
    
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))


    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 5)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'example_config_file')
    run(config_path)
