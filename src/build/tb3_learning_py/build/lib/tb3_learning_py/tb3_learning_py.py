#!/usr/bin/python
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray, Int32, Bool

#install for multiple node executor
from rclpy.executors import SingleThreadedExecutor

import json
import sys
#from . import qtable_model2, qtable_learn_py
import time
import numpy as np
import io, re, os, random
import argparse
from collections import deque
from collections import namedtuple
import scipy.stats
import tensorflow as tf
from tensorflow import keras as K
from PIL import Image
from tensorflow.python.client import device_lib
import csv
import copy

#for explain
import types


from dqnagent import DeepQNetworkTrainer, DeepQNetworkAgent, DeepQNetworkAgentTest
from fn_framework import FNAgent, Trainer, Logger



#print('abspath:     ', os.path.abspath(__file__))

#TODO : 一旦はグローバル変数で宣言するが、ゆくゆくはパラメタで受け取る
robo = "TB3RoboModel"

#model.load('./model/dqntable_model.csv')

DIS_LENGTH = 24
#スタート地点の座標、手入力
initial_tf = [0.0, 0.0, 0.0]
ACTION_NUM = 5 #TODO;正しく設定
goal_tf = [1.0, 1.0, 1.0]
stopped = False
is_test = False
file_name = "dqn_agent.h5" if not is_test else "dqn_agent_test.h5"
Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

done = False
last_state = np.array([0. for i in range(28)])
last_action = 0

last_tf = np.array([0.2, 0.2, 0.0, 90])
#x, y, z, Unityでのy軸周りの回転角
current_tf = np.array([0.2, 0.2, 0.0, 90])
dis_list = np.ones(360)
v = 0.05
tfs = []



count_log = []
action_log = [0 for i in range(5)]
goal_log = []
is_goal = False

#環境を初期化するためのフラグ。False→Trueへ更新の責任はaction_publisherにあり、True→Falseへの更新の責任はdone_subscriberにある。
global_done = False

#環境初期化フラグdoneを変更する権限が、本ノードにあるかUnity側にあるかを制御するフラグ
is_done_changed_by_thisnode = False

#総実行回数
all_steps = 0

learning_stop = False

    
class DistanceSubscriber(Node):
    def __init__(self):
        super().__init__('distance_subscriber')
        self.sub = self.create_subscription(
            Float64MultiArray,
            'scan',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global dis_list
        for i in range(360):
            dis_list[i] = msg.data[i]
        #print(state[0])

class TfSubscriber(Node):
    def __init__(self):
        super().__init__('tf_subscriber')
        self.sub = self.create_subscription(
            Float64MultiArray,
            'current_tf',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global current_tf
        for i in range(4):
            current_tf[i] = msg.data[i]
        #print(state[0])

class DoneSubscriber(Node):
    def __init__(self):
        super().__init__('done_subscriber')
        self.sub = self.create_subscription(
            Bool,
            'done',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global global_done, is_done_changed_by_thisnode
        #このノードは終了したと言っているがトピックは終了していないと主張している場合
        #Unity側からの書き換えが行われたということなので、
        #is_done_changed_by_thisnode を更新し、global_doneをFalseにしてActionPublisherにエピソード再開を許可
        if not msg.data and global_done:
            is_done_changed_by_thisnode = False
            global_done = False

class DonePublisher(Node):
    def __init__(self):
        super().__init__('done_publisher')
        self.publisher_ = self.create_publisher(Bool, 'done', 1)
    
    def timer_callback(self):
        global global_done, is_done_changed_by_thisnode
        #Unity側の環境がrosのdoneトピックを変更しており、かつTrueのとき送信
        #具体的には、エピソードを終了する旨を全体に共有する操作。一回のみ実行し、Unity側がdoneトピックを書き換えるのを待つ
        if global_done and not is_done_changed_by_thisnode:
            done_msg = Bool()
            done_msg.data = True
            self.publisher_.publish(done_msg)
            self.get_logger().info('Publishing global done: "%d"' % done_msg.data)
            is_done_changed_by_thisnode = True

        #Unity側の終了処理待ち
        elif global_done and is_done_changed_by_thisnode:
            print("done publisher is waiting for Unity\n")
        
        #Unity側がrosのdoneトピックを変更し、それをdonesubscriberが検知してglobal_doneをFalseに変更してくれた場合
        #Donesubscriberで、次回の変更に備えて is_done_changed_by_thisnode が更新される。


#実機を考えるとgeometoryで送って方がいいかも？　データ形式は多分Twist
class ActionPublisher(Node):

    def __init__(self):
        global is_test

        super().__init__('action_publisher')
        self.publisher_ = self.create_publisher(Int32, 'cmd_dir', 1)

        #subscribe:state
        #self.sub = Node.create_subscription(Float64MultiArray, 'min_arr', self.listener_callback, qos_profile=custom_qos_profile)
        timer_period = 3.0 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        #about learning
        
        #epsilonの初期値と最終値はTrainerクラスの内部で設定され、episode_endごとに減衰する設計になっている。
        #GAMMAはTrainerクラスの内部で0.99に設定されている。

        #train関数に対応
        #self.trainer.train(obs, test_mode=is_test)
        self.actions = list(range(ACTION_NUM))

        self.trainer = DeepQNetworkTrainer(file_name=file_name)
        self.path = self.trainer.logger.path_of(self.trainer.file_name)
        self.agent = DeepQNetworkAgent(1.0, self.actions)

        #ここから追加した変数 
        self.buffer_size = 2048

        #現在何エピソード目か
        self.episode_count = 0

        #stepに対してobserveごとに確認している。frameの仕組みがわかればなんとかなりそう。基本的に描画周りなのであまり使わなそう。
        self.frames = []

        #エピソード完了フラグ
        self.done = False

        #再試行処理完了フラグ。一度のみ実行するために利用。必ず本クラスの内部で更新すること
        self._restart_done = False

        #何ステップ目か
        self.step_count = 0

        #トレーニングができたエピソードの回数。この回数をもとに教師モデルを更新する。
        self.trainer.training_count = 0

        #学習につかったExperience(遷移前の状態、行動、報酬、遷移後の状態、完了したかどうか)を記録している。
        #バッファ分だけExperienceをすべて記録し、それらを一括で学習に用いる。
        self.trainer.experiences = deque(maxlen=self.buffer_size)

        #合計報酬のログ(per episode)
        self.trainer.reward_log = []

        #lossの初期値を設定するだけ
        self.trainer.episode_begin(self.episode_count, self.agent)

        #既存のモデルを用いて実行する際のModelのloadは未実装。
        #通常はagent.initialize(self, experiences, optimizer)を呼んだ際にloadされる。
        #具体的には、Trainerのbegin_train実行時に呼ばれている。その際、experiences[0]のstateの形を見て入力層を作る。


        self.goal_count = 0

    def timer_callback(self):
        global global_done, all_steps, learning_stop

        #TODO:繰り返し回数の設定
        if all_steps > 20000:
            learning_stop = True
            self.agent.model.save(self.path)
            print("learning done")

        if not learning_stop:

            #既存コードのtrain_loopに対応。
            all_steps += 1
            if self.all_steps % 100 == 0:
                print("now : ", all_steps, " step")

            #TODO:適切な位置に配置する
            last_tf = current_tf

            if not self.done:
                best_action = self.best_action_learning()
                action_log[best_action] += 1
                send_msg = Int32()
                
                send_msg.data = best_action
                self.publisher_.publish(send_msg)
                self.get_logger().info('Publishing: "%d"' % send_msg.data)

                #step_countの更新はbest_action_learning関数内で行う

            elif self.done:
                self.do_ifdone()

    def msg2state(self, given_dis_list, given_current_tf, given_goal_tf):
        global is_goal

        DIS_LENGTH = 24
        given_dis_list = np.array(given_dis_list)
        given_current_tf = np.array(given_current_tf)
        given_goal_tf = np.array(given_goal_tf)
        #dis_list : 距離センサの値[m], 360度
        #current_tf : 現在地点の座標と向き　[x, y, z, z軸周りの回転角度(反時計回り正)]
        #goal_tf : 目標地点の座標 [x, y, z]
        #これを、距離24次元 + 目標地点までの距離[m] + 目標地点までの向き[rad, -pi~pi] + 最も近い障害物までの距離[m] + 最も近い障害物までの方向[rad, -pi~pi]に変換したい
        state = np.array([0.0 for i in range(DIS_LENGTH + 4)])
        
        #距離24次元については、とりあえず15°ずつの平均値を用いる。一部測定値が0の部分がある(センサの測定レンジ以外)ので、その部分は省いて平均を取る。
        np_dis_list = np.array(given_dis_list)
        for i in range(DIS_LENGTH):
            target_dis_list = np_dis_list[i*15 : (i+1) * 15]
            #非ゼロの平均値を取得
            state[i] = np.mean(target_dis_list[np.nonzero(target_dis_list)])
        
        #目標地点までの二次元ベクトル
        target_vector = given_goal_tf[0:2] - given_current_tf[0:2]

        #目標地点までの距離[m]
        dis2goal = np.linalg.norm(target_vector)

        #目標地点までの角度の差 rad_dif[rad]
        #引数がarctan2(y, x)なので注意
        rad_dif = (np.arctan2(target_vector[1], target_vector[0]) - \
                              np.deg2rad(given_current_tf[3])) % (2 * np.pi)
        
        if rad_dif > np.pi:
            rad_dif =  rad_dif - 2 * np.pi
        
        #最も近い障害物までの距離[m]と方向[rad] : 
        dis2obs = np.min(np_dis_list[np.nonzero(np_dis_list)])
        minimum_obstacle_index = np.where(np_dis_list==dis2obs)[0][0]
        obs_rad_dif = minimum_obstacle_index * np.pi / 180.
        if obs_rad_dif > np.pi:
            obs_rad_dif = obs_rad_dif - 2 * np.pi
        
        state[0:DIS_LENGTH]=scipy.stats.zscore(state[0:DIS_LENGTH])
        dis_ndarray = np.array([dis2goal, dis2obs])
        arg_ndarray = np.array([rad_dif, obs_rad_dif])
        #state[DIS_LENGTH], state[DIS_LENGTH+2]= dis_ndarray / max(np.linalg.norm(dis_ndarray), 0.001)
        #state[DIS_LENGTH+1], state[DIS_LENGTH+3]= arg_ndarray / max(np.linalg.norm(arg_ndarray), 0.001)
        state[DIS_LENGTH], state[DIS_LENGTH+2]= dis_ndarray
        state[DIS_LENGTH+1], state[DIS_LENGTH+3]= arg_ndarray / 10.
        #print(state)



        

        reward = 0
        #距離に対する報酬, Dg:初期の距離
        Dg = np.linalg.norm(given_goal_tf[0:2] - initial_tf[0:2])
        reward += 10 * max(-1, 1 - np.linalg.norm(target_vector) / Dg)
        #print(reward)

        #角度に対する報酬
        
        #reward += (1 - 2 * abs(rad_dif / np.pi))

        #回転が連続することを防ぐ
        '''
        if len(trainer.experiences) > 10:
            recent_actions = [e.a for e in trainer.get_recent(8)]
            #print(recent_actions)
            if min(recent_actions) > 0:
                reward -= 4
        '''
  
  
        #ある程度エピソードが進んだ場合は距離と角度報酬を低減
        #reward = reward * 10 / (episode_count // 100 + 10)
        #reward = 0

        if self.step_count > 200:
            print("too long episode detected")
            current_tf[0] = 0.2
            current_tf[1] = 0.2
            #reward -= 50
            done = True


        # collision : 0.2m以下とする
        self.done = False
        if np.min(np_dis_list[np.nonzero(np_dis_list)]) < 0.2:
            print("!collision!")
            reward -= 30 * (1. - 1./(self.goal_count // 10 + 2))
            done = True
        
        #ゴール判定を初期は広げるというアプローチ。これ自体は悪くないがやや目的とずれる気もするので保留
        #goal_rad = 0.03 + 10. / (goal_count + 10.)
        goal_rad = 0.2
        if np.linalg.norm(target_vector) < goal_rad :
            print("!goal!")
            reward += 500
            done = True
            goal_count += 1
            is_goal = True

        return state, reward, done

    def best_action_learning(self, observe_interval=10, buffer_size=1024):
        global last_action, last_state, stopped
        global current_tf, goal_tf, dis_list
        #結果がわかっているlast_actionについて学習を行っていることに注意。

        #get last state
        s = last_state
        a = last_action
        

        # ここでaction
        # aを行ったあとの状態を記録
        n_state, reward, done = self.msg2state(dis_list, current_tf, goal_tf)
        e = Experience(s, a, reward, n_state, done)
        self.trainer.experiences.append(e)

        #ある程度データが溜まったらそれ以降学習開始
        if not self.trainer.training and len(self.trainer.experiences) > buffer_size:
            #モデルを設定するなどなんやかんややっている。logger周り以外はなんとかなっていそうだが...
            self.trainer.begin_train(self.episode_count, self.agent)
            self.trainer.training = True

        #ここでbatchを用いた学習を行う
        #多分大丈夫だと思うけどなあ。。。
        self.trainer.step(self.episode_count, self.step_count, self.agent, e)
        self.step_count += 1

        #ここは実装済み
        best_action = self.agent.policy(n_state)
        last_state = n_state
        last_action = best_action

        return best_action
    
    def do_ifdone(self):
        global global_done, count_log, is_goal

        #グローバルに終了処理が行われていない場合に一回のみ実行。
        if not global_done and not self._restart_done:
            #エピソードが終わったことをglobalに示すフラグをpublish
            global_done = True
            #agent._teacher_modelを使っていることに注意。あとloggerも使っている。それ以外は不安要素はなさそう
            que_length = len(self.trainer.current_experiences)

            if is_goal:
                for i in range(que_length):
                    #trainer.positive_experiences.popleft()
                    self.trainer.positive_experiences.append(self.trainer.current_experiences.popleft())
            else:
                for i in range(que_length):
                    #trainer.negative_experiences.popleft()
                    self.trainer.negative_experiences.append(self.trainer.current_experiences.popleft())
            copy_p_deque = copy.deepcopy(self.trainer.positive_experiences)
            copy_n_deque = copy.deepcopy(self.trainer.negative_experiences)
            copy_end_deque = copy.deepcopy(self.trainer.end_experiences)

            #experiencesにキューを追加
            #positive_experiences, negative_experiencesはそれぞれ保存されているキュー

            for i in range(2048):
                #終了時点のデータを忘れないようにする
                if len(copy_end_deque) != 0:
                    self.trainer.experiences.append(copy_end_deque.pop())
                else:
                    if len(copy_p_deque) == 0:
                        if len(copy_n_deque) != 0:
                            self.trainer.experiences.append(copy_n_deque.pop())
                    else:
                        self.trainer.experiences.append(copy_p_deque.pop())
            print("all:", len(self.trainer.experiences))
            print("end:", len(self.trainer.end_experiences))
            print("p:", len(self.trainer.positive_experiences))
            print("n:", len(self.trainer.negative_experiences))

            self.trainer.episode_end(self.episode_count, self.step_count, self.agent, count_log, is_goal, self.goal_count)

            #restartを複数回実行しないためにクラスで責任をもって更新
            self._restart_done = True
        
        #グローバルに終了処理を行っている最中で、まだ処理が終わっていない状況であるので待つ。
        elif global_done:
            print("waiting for restart...\n")

        #グローバルな終了処理が完了し、かつリスタート準備もできているのでエピソードを再開する。
        elif not global_done and self._restart_done: 
            self.step_count = 0
            print("goal_count", self.goal_count)
            count_log.append(self.step_count)
            goal_log.append(self.goal_count)
            episode_count += 1
            is_goal = False

            #エピソード数を加算
            self.episode_count += 1
            self._restart_done = False
            self.done = False
            self.trainer.episode_begin(self.episode_count, self.agent)
        



def main(args=None):
    rclpy.init(args=args)
    exec = SingleThreadedExecutor()
    action_publisher = ActionPublisher()
    done_publisher = DonePublisher()
    distance_subscriber =  DistanceSubscriber()
    tf_subscriber =  TfSubscriber()
    exec.add_node(action_publisher)
    exec.add_node(done_publisher)
    exec.add_node(distance_subscriber)
    exec.add_node(tf_subscriber)
    exec.spin()
    #rclpy.spin(action_publisher)
    #rclpy.spin(state_subscriber)
    exec.destroy_node()
    exec.shutdown()


if __name__ == '__main__':
    main()