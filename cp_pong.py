#!/usr/bin/env python3

from hee_human_trainer import *
import numpy as cp
#import numpy as np
import pickle
import gym
import argparse
import json
import os
import csv

GRID_SIZE = 80 * 80
LEARNING_RATE = 1e-4
GAMMA = 0.99
DECAY_RATE = 0.99 

# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--render" , help="show the game screen", action='store_true')
parser.add_argument("--seed" , help="set the seed", type=int , default=1)
parser.add_argument("--save_every" , help="how often to save moves and models", type=int, default=5000)
parser.add_argument("--save_activations_every" , help="how often to save activations", type=int, default=5000)
parser.add_argument("--gpu" , help="GPU id to use with cupy", type=int, default=0)   

#Training parameters
parser.add_argument("--episodes" , help="train for episodes", type=int, default=100000)
parser.add_argument("--no_train" , help="turn off network updates", action='store_true')
parser.add_argument("--retrain" , help="path to folder to retrain from", type=int, default=0)
parser.add_argument("--retrain_start" , help="starting episode to retrain from")
parser.add_argument("--pickle" , help="use to retrain without replay")

# Neural Network
parser.add_argument("--hidden" , help="number of hidden nodes" , type=int, default=200)
parser.add_argument("--batch_size" , help="episodes to batch", type=int , default=10)
parser.add_argument("--dropout" , help="dropout probability", type=float , default=0.)
parser.add_argument("--normalize" , help="normalize neurons during forward pass", action='store_true')
                                                                         
# Human Trainer Parameters
parser.add_argument("--use_trainer" , help="use the human trainer", action="store_true")
parser.add_argument("--intensity" , help="controls the multiplicative effect of the teacher. Default 0.15", 
                                    type=float, default=0.15 )
parser.add_argument("--numrange" , help="sets scan range of the teacher. Default 5" , type=int, default=5 )
parser.add_argument("--decay" , help="sets episode to linearly decay to 0 at. No decay by default" , 
                                type=float, default=0 )
args = parser.parse_args()

# Set the GPU to use
#cp.cuda.Device(args.gpu).use()

if args.no_train: print('WARNING. THIS WILL RUN WITHOUT TRAINING!!!')

# Folders, files, metadata start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH = "ht" if args.use_trainer else "no_ht"
PATH = PATH + "-" + str(args.episodes) + "-S" + str(args.seed) + "-H" + str(args.hidden)
MODEL_NAME =  PATH + "/pickles/"
ACTIVATIONS = PATH + "/activations/"
STATS = PATH + "/stats.csv"
MOVES = PATH + "/moves.csv"

os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)
os.makedirs(os.path.dirname(ACTIVATIONS), exist_ok=True)
print('Saving to: ' + PATH)

with open(PATH+'/metadata.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
# Folders, files, metadata end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Model initialization start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cp.random.seed(args.seed)

model = {}
model['W1'] = cp.random.randn(args.hidden, GRID_SIZE) / cp.sqrt(GRID_SIZE)
model['W2'] = cp.random.randn(args.hidden) / cp.sqrt(args.hidden)

if args.pickle:
    print('Opening Pickle file')
    model = pickle.load(open(args.pickle, 'rb'))
elif args.retrain:
    print('Retraining from ', args.retrain)
    model = pickle.load(open(args.retrain + '/pickles/' + 
                             args.retrain_start  + '.p', 'rb'))

grad_buffer = {k: cp.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: cp.zeros_like(v) for k, v in model.items()}
# Model initialization end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Functions start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_csv(data, filename):
    with open(filename, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()

# Karpathy
def sigmoid(value):
    """Activation function used at the output of the neural network."""
    return 1.0 / (1.0 + cp.exp(-value)) 
    
# Karpathy
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward. """
    discounted_r = cp.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0

        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    discounted_r -= cp.mean(discounted_r)
    discounted_r /= cp.std(discounted_r)
    return discounted_r

# Karpathy with added normalization and dropout options
def policy_forward(screen_input):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
    int_h = cp.dot(model['W1'], screen_input)
    
    # Added
    if args.normalize:
        mean = cp.mean(int_h)
        variance = cp.mean((int_h - mean) ** 2)
        int_h = (int_h - mean) * 1.0 / cp.sqrt(variance + 1e-5)
    
    int_h[int_h < 0] = 0  # ReLU nonlinearity
    
    # Added
    if args.dropout != 0:
        mask = cp.random.binomial(1, 1-args.dropout) * (1.0/(1-args.dropout))
        int_h = int_h * mask
        
    logp = cp.dot(model['W2'], int_h)
    p = sigmoid(logp)
    return p, int_h  # return probability of taking action 2, and hidden state

# Karpathy
def policy_backward(int_harray, grad_array):
    """ backward pass. (int_harray is an array of intermediate hidden states) """
    delta_w2 = cp.dot(int_harray.T, grad_array).ravel()
    delta_h = cp.outer(grad_array, model['W2'])
    delta_h[int_harray <= 0] = 0  # backprop relu
    delta_w1 = cp.dot(delta_h.T, epx)
    return {'W1': delta_w1, 'W2': delta_w2}
    
# Functions end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
if __name__ == "__main__":
    env = gym.make("Pong-v0")
    env.seed(args.seed)
    observation = env.reset()
    cp.random.seed(args.seed)

    prev_x = None 
    running_reward = None
    proximity = False
    ep_num = 1
    reward_sum = 0
    check = 0 ; cycle = 0 ; p_mod = 0
    current_horizon = [list(), list(), list(), list(), list()]
    xs, ys, hs, dlogps, drs = [], [], [], [], []
    stats_buffer, track_activations, moves_buffer = [], [], []

    # Replay games start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """ The purpose here is to generate reproducible results if we have to
    retrain agents. This can take some time. If reproducible results aren't 
    required then load the model with --pickle"""
    if args.retrain:
        with open(args.retrain + "/" + args.retrain + "-moves.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                moves_buffer.append(line)

        rmsprop_cache = pickle.load(open(args.retrain + '/rmsprop_cache.p', 'rb'))
        i = 0
        end = int(args.retrain_start)
        while ep_num <= end:
            action = moves_buffer[ep_num-1][i]
            action = 2 if action == '1' else 3        
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            cp.random.uniform()
            i += 1
            if done:
                observation = env.reset()
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                reward_sum = 0
                ep_num += 1 
                i = 0
        moves_buffer = []
    # Replay games end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # Training start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while ep_num <= args.episodes:
        if args.render: env.render()

        cur_x, current_horizon = screen_process(observation, current_horizon, cycle, args.numrange)
        move_recommendation = follow_the_ball(current_horizon) if args.use_trainer else 0

        x = cur_x - prev_x if prev_x is not None else cp.zeros(GRID_SIZE)
        x = cp.array(x)
        prev_x = cur_x
        
        prob_up, h = policy_forward(x)
        
        if ep_num % args.save_activations_every == 0:
            track_activations.append(h)

        decay_mod = args.intensity if args.decay == 0 else args.intensity * (args.decay ** int(ep_num / args.batch_size))

        p_mod = h_diff_mod(prob_up, move_recommendation, weight=decay_mod)

        action = 2 if cp.random.uniform() < prob_up + p_mod else 3
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        
        # record various intermediates (needed later for backprop)
        
        action = 1 if action == 2 else 0
    
        xs.append(x)  ; ys.append(action) ; hs.append(h) ; drs.append(reward)
        dlogps.append(action - (prob_up + p_mod) )
        
        cycle += 1
        
        if done:  # an episode finished
            moves_buffer.append(ys)
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            
            epx = cp.vstack(xs)     #screen 
            eph = cp.vstack(hs)     #hidden layer 
            epr = cp.vstack(drs)    #reward
            epdlogp = cp.vstack(dlogps) 
            xs, hs, drs, ys, dlogps = [], [], [], [], []
            
            stats_buffer.append((ep_num, reward_sum, running_reward))
            reward_sum = 0; observation = env.reset()
            
            if ep_num % args.save_activations_every == 0:
                pickle.dump(track_activations, open(ACTIVATIONS  + str(ep_num) + '.p', 'wb'))
                track_activations = []
                
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, GAMMA)
            discounted_epr -= cp.mean(discounted_epr)
            discounted_epr /= cp.std(discounted_epr)
            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes. Default 10.
            if ep_num % args.batch_size == 0:
            
                w1_before = model['W1']
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g ** 2
                    model[k] += LEARNING_RATE * g / (cp.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = cp.zeros_like(v)  # reset batch gradient buffer
                pickle.dump(rmsprop_cache, open(PATH+'/rmsprop_cache.p', 'wb'))
            
            if ep_num % args.save_every == 0:
                pickle.dump(model, open(MODEL_NAME  + str(ep_num) + '.p', 'wb'))
                save_csv(stats_buffer, STATS); stats_buffer = []
                save_csv(moves_buffer, MOVES); moves_buffer = []
                
            prev_x = None
            check = 0
            ep_num += 1

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            current_horizon = list()
            cycle = 0
    # Training end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
