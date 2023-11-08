import tensorflow as tf
from keras import layers
import keras
import numpy as np
import threading
from tensorflow.python.keras.backend import set_session
import multiprocessing
from my_environment import RouteEnvironment
import os
import pandas as pd
import openpyxl
from process_data import get_route_data
from matplotlib import pyplot as plt

# routes = pd.read_excel("data/processed_data.xlsx",engine='openpyxl')
#data =   get_route_data([routes.loc[0]['first_latitude'],routes.loc[0]['first_longitude']],[routes.loc[0]['last_latitude'],routes.loc[0]['last_longitude']],routes.loc[0]['first_max_speed'])
data = pd.read_excel("first_route_normalized_new.xlsx",engine='openpyxl',dtype={'current_speed': float,'current_speed_normalized': float})
# data = data.drop_duplicates()
# data = data.reset_index()[['distance','distance_normalized','acceleration_to_slope','acceleration_to_slope_normalized','max_speed','max_speed_normalized','speed']]

# print(data)
# Global variables
max_global_episodes = 30000

# configure Keras and TensorFlow sessions and graph
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
graph = tf.compat.v1.get_default_graph()

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(400, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(400, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    #inputs =  tf.compat.v1.convert_to_tensor(inputs,dtype=tf.float32)
    x = self.dense1(np.array(inputs).reshape(-1,7))
    logits = self.policy_logits(x)
    v1 = self.dense2(np.array(inputs).reshape(-1,7))
    values = self.values(v1)
    return logits, values
  
class MasterAgent():
    rewards = []
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001,use_locking=True)
        print(self.state_size,self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size) #global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        workers = [Worker(self.state_size,self.action_size,self.global_model,self.optimizer,i) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        
        [w.join() for w in workers]
        for i, worker in enumerate(workers):
            worker.plot()
        self.plot()

    def plot(self):
        mov_avg = []
        window_size = 50
        for i in range(len(MasterAgent.rewards)-window_size+1):
            window = MasterAgent.rewards[i:i+window_size]
            average = sum(window)/window_size
            mov_avg.append(average)
        plt.plot(mov_avg)
        plt.title('Mov Avg Rewards Master')
        plt.savefig('results/4/rewards_30000_eps.png')
        plt.show()

class Memory():
    def __init__(self):
        self.states = []
        self.action = []
        self.rewards = []
    def store(self,state,action,reward):
        self.states.append(state)
        self.action.append(action)
        self.rewards.append(reward)
    def clear(self):
        self.states = []
        self.action = []
        self.rewards = []

class Worker(threading.Thread):
    global_episode = 0
    global_moving_avg_reward = 0
    best_score = 0
    save_lock = threading.Lock()
    def __init__(self,state_size,action_size,global_model,optimizer,idx,save_dir='results/4'):
        super(Worker,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = global_model
        self.env = RouteEnvironment(data)
        self.optimizer = optimizer
        self.local_model = ActorCriticModel(self.state_size,self.action_size)
        self.index = idx
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.ep_reward = 0.
        self.rewards = []
        self.gamma = 0.99

    def run(self):
        total_step = 1
        mem = Memory()
        best_score_mem = Memory()
        while Worker.global_episode < max_global_episodes:
            current_state = self.env.reset()
            mem.clear()
            best_score_mem.clear()
            self.ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            steps_count = 0

            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.compat.v1.convert_to_tensor(current_state,dtype=tf.float32))
                probs = tf.nn.log_softmax(logits)
                action = np.random.choice(self.action_size, p=np.exp(probs.numpy()[0]))

                current_state, new_state, reward, done = self.env.step(action)
                self.ep_reward += reward
                mem.store(current_state, action, reward)
                best_score_mem.store(current_state, action, reward)
                MasterAgent.rewards.append(reward)
                # self.states.append(current_state)
                steps_count +=1

                if steps_count == 15 or done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,new_state,mem,self.gamma)
                    self.ep_loss += total_loss
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads,self.global_model.trainable_weights))
                    self.local_model.set_weights(self.global_model.get_weights())
                    mem.clear()
                    
                    if done:
                        Worker.global_moving_avg_reward = \
                            print(Worker.global_episode, self.ep_reward, Worker.best_score, self.index, self.ep_loss, ep_steps)
                        Worker.global_episode += 1
                        if self.ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, episode score: {}".format(self.save_dir,self.ep_reward))
                                self.global_model.save_weights(os.path.join(self.save_dir,
                                    'model_{}.h5'.format("route")))
                                Worker.best_score = self.ep_reward
                            best_score_df = pd.DataFrame(best_score_mem.states)
                            best_score_df['actions'] = best_score_mem.action
                            best_score_df['reward'] = best_score_mem.rewards
                            best_score_df.to_excel("results/4/best_model_actions.xlsx")
                ep_steps += 1
                current_state = new_state
                total_step += 1
            
            self.rewards.append(self.ep_reward)
    def plot(self):
        mov_avg = []
        window_size = 10
        for i in range(len(self.rewards)-window_size+1):
            window = self.rewards[i:i+window_size]
            average = sum(window)/window_size
            mov_avg.append(average)
        plt.plot(mov_avg)
        plt.title('Mov Avg Rewards Worker {}'.format(str(self.index)))
        plt.savefig(self.save_dir+'/rewards_30000_eps_Worker {}.png'.format(str(self.index)))
        plt.show()
        
    
    def compute_loss(self,done,new_state,mem,gamma,temperature=1):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state,
                                dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in mem.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(mem.states),
                                dtype=tf.float32))

        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards),
                                dtype=tf.float32) - values
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(mem.action, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits/temperature)
        entropy = tf.reduce_sum(policy * tf.compat.v1.log(policy + 1e-20), axis=1)
        
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot,
                                                                logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
    
    
STATE_SIZE = 7
ACTION_SIZE = 5


master = MasterAgent(STATE_SIZE,ACTION_SIZE)
master.train()