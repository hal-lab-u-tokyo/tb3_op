import random
import argparse
from collections import deque
import tensorflow as tf
from tensorflow import keras as K
from PIL import Image

from . import fn_framework as FF
#from fn_framework import FNAgent, Trainer

import csv
import os
import numpy as np

FNAgent = FF.FNAgent
Trainer = FF.Trainer
Logger = FF.Logger

class DeepQNetworkAgent(FNAgent):
    #Q 学習機本体
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None
        csv_path = os.getcwd() + "/dqnmodel.csv"
        
        if not os.path.isfile(csv_path):
            with open(csv_path, 'w') as f:
                f.write('')
        self.modelcsv = "./dqnmodel.csv"

        # with open(self.modelcsv, 'r') as f:
        #     reader = csv.reader(f)
        #     l = [row for row in reader]

        # print(l)
    
    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def model_reset(self):
         with open(self.modelcsv, 'w') as f:
                f.write('')

    def dis2state(self, distance_array, tf, destination_tf):
        #距離センサの値を正規化する必要がありそう。
        state = distance_array + tf + destination_tf
        return state
    
    #ここから書籍の引用

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()

        #NNの構造定義
        model.add(K.layers.Dense(64, input_shape = feature_shape, activation='relu'))
        model.add(K.layers.Dropout(0.2))
        model.add(K.layers.Dense(5, activation='linear', kernel_initializer=normal))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

    #各行動の確率
    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]
    

    def update(self, experiences, gamma):
        #e.s ：　現在の状態
        #e.n_s : 遷移先の状態
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        #future : 各行動によって得られると想定される報酬
        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            #dは終了フラグ falseで実行
            #報酬の計算
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())

#テスト用のエージェント
class DeepQNetworkAgentTest(DeepQNetworkAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal,
                                 activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

class DeepQNetworkTrainer(Trainer):

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.99, initial_epsilon=0.5, final_epsilon=1e-3,
                 learning_rate=1e-3, teacher_update_freq=3, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10
        self.reward_log = []
        self.loss_log = []
        self.epsilon_log = []

    def train(self, env, episode_count=1200, initial_count=200,
              test_mode=False, render=False, observe_interval=100):
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = DeepQNetworkAgent(1.0, actions)
        else:
            agent = DeepQNetworkAgentTest(1.0, actions)
            observe_interval = 0
        self.training_episode = episode_count

        self.train_loop(env, agent, episode_count, initial_count, render,
                        observe_interval)
        return agent

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)
        #self.logger.callback.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    def episode_end(self, episode_count, step_count, agent, count_log, is_goal, goal_count):
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.loss = self.loss / step_count
        self.reward_log.append(reward)
        self.loss_log.append(self.loss)

        if self.training and self.is_event(self.training_count, self.teacher_update_freq):
            #print("teacher!!!!!")
            agent.update_teacher()

        #diff = (self.initial_epsilon - self.final_epsilon)
        #decay = diff / self.training_episode
        #decay = diff / episode_count
        #print(agent.epsilon)
        
        if is_goal:
            #decay = diff / 100.
            #agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)
            agent.epsilon = agent.epsilon * 0.99

        #初めてゴールするまでは1.0なのでそれを排除(みやすさのため)
        if goal_count > 0:
            self.epsilon_log.append(agent.epsilon)
    ##ここから既存のQ学習    

    """
    def select_q(self, state, act):
        #状態とアクションをキーに、q 値取得
        if ((state, act) in self._values.keys()):
            #print("happy")
            return self._values[(state, act)]
        else:
            # Q 値が未学習の状況なら、Q 初期値
            # print(self._values.keys())
            self._values[(state, act)] = self._initial_q
            return self._initial_q

    def save(self):
        with open(self.modelcsv, 'w') as f:
            writer = csv.DictWriter(f, self._values.keys())
            writer.writeheader()
            writer.writerow(self._values)
            
    def load(self):
        with open(self.modelcsv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    tpl_key = eval(key)
                    self._values[tpl_key] = float(value)
        # for key, value in self._values.items():
        #     print(type(value))


    def set(self, state, act, q_value):
        #Q 値設定
        self._values[(state, act)] = q_value

    def learning(self, state, act, max_q):
        #Q 値更新
        pQ = self.select_q(state, act)
        new_q = pQ + self._alpha * (self.score_list[act] + self._gamma * max_q - pQ)
        #print(new_q - pQ)
        self.set(state, act, new_q)

    def add_fee(self, state, act, fee):
        #報酬を与える
        pQ = self.select_q(state, act)
        new_q = pQ + self._alpha * (fee - pQ)
        self.set(state, act, new_q)
    
    """