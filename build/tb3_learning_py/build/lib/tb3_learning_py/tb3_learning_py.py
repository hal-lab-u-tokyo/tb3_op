#!/usr/bin/python
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray, Int32, Bool, UInt32

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3

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


#from dqnagent import DeepQNetworkTrainer, DeepQNetworkAgent, DeepQNetworkAgentTest
#from fn_framework import FNAgent, Trainer, Logger

from . import dqnagent
from . import fn_framework as FF

DeepQNetworkTrainer = dqnagent.DeepQNetworkTrainer
DeepQNetworkAgent = dqnagent.DeepQNetworkAgent
DeepQNetworkAgentTest = dqnagent.DeepQNetworkAgentTest
FNAgent = FF.FNAgent
Trainer = FF.Trainer


#print('abspath:     ', os.path.abspath(__file__))

#TODO : 一旦はグローバル変数で宣言するが、ゆくゆくはパラメタで受け取る
robo = "TB3RoboModel"

#model.load('./model/dqntable_model.csv')

DIS_LENGTH = 24
#スタート地点の座標、手入力
initial_tf = [0.0, 0.0, 0.0]
ACTION_NUM = 5 #TODO;正しく設定
goal_tf = [8.0, 8.0, 0.0]
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
goal_percentage_log = []
is_goal = False

#制御側の状態のフラグ。0:環境リセット後待機中、動作中：1、終了判定かつ環境リセット前、2
# UnityのSimStartに合わせている。
WAITSTART = 0
WAITSTOP = 1
WAITRESET = 2

sim_state_flag = WAITSTOP

#Unityの状態のフラグ。0:環境リセット後待機中、動作中：1、終了判定かつ環境リセット前、2
robot_state_flag = WAITSTART


#総実行回数
all_steps = 0

learning_stop = False

    
class DistanceSubscriber(Node):
    def __init__(self):
        super().__init__('distance_subscriber')
        self.sub = self.create_subscription(
            Float64MultiArray,
            'dis_arr',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global dis_list
        for i in range(360):
            dis_list[i] = msg.data[i]
        #print(state[0])

class PositionSubscriber(Node):
    def __init__(self):
        super().__init__('position_subscriber')
        self.sub = self.create_subscription(
            Vector3,
            'TB3RoboModel_robot_position',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global current_tf
        current_tf[0] = msg.x
        current_tf[1] = msg.z
        current_tf[2] = msg.y
        #print(state[0])

class TfSubscriber(Node):
    def __init__(self):
        super().__init__('tf_subscriber')
        self.sub = self.create_subscription(
            TFMessage,
            'TB3RoboModel_tf',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global current_tf
        current_tf[3] = 180 * msg.transforms[0].transform.rotation.z
        #self.get_logger().info('Robot angle: "%d"' % current_tf[3])
        #print(state[0])

class RobotStateSubscriber(Node):
    def __init__(self):
        super().__init__('state_subscriber')
        self.sub = self.create_subscription(
            UInt32,
            'TB3RoboModel_robot_state',
            self.listener_callback,
            10)
        self.sub # prevent unused variable warning

    def listener_callback(self, msg):
        global robot_state_flag
        #このノードは終了したと言っているがトピックは終了していないと主張している場合
        #Unity側からの書き換えが行われたということなので、
        #is_done_changed_by_thisnode を更新し、global_doneをFalseにしてActionPublisherにエピソード再開を許可
        robot_state_flag = msg.data
        #self.get_logger().info('Robot state: "%d"' % msg.data)


class SimStatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')
        self.publisher_ = self.create_publisher(UInt32, 'TB3RoboModel_sim_state', 1)
        self.timer_period = 1.0 # seconds
        self.state_timer = self.create_timer(self.timer_period, self.state_timer_callback)
    
    def state_timer_callback(self):
        global sim_state_flag
        done_msg = UInt32()
        done_msg.data = sim_state_flag
        self.publisher_.publish(done_msg)
        self.get_logger().info('Sim state: "%d"' % done_msg.data)


#実機を考えるとgeometoryで送った方がいいかも？　データ形式は多分Twist
class ActionPublisher(Node):

    def __init__(self):
        global is_test

        super().__init__('action_publisher')
        self.publisher_ = self.create_publisher(Int32, 'cmd_dir', 1)

        #subscribe:state
        #self.sub = Node.create_subscription(Float64MultiArray, 'min_arr', self.listener_callback, qos_profile=custom_qos_profile)
        self.timer_period = 1.0 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        #about learning
        
        #epsilonの初期値と最終値はTrainerクラスの内部で設定され、episode_endごとに減衰する設計になっている。
        #GAMMAはTrainerクラスの内部で0.99に設定されている。

        #train関数に対応
        #self.trainer.train(obs, test_mode=is_test)
        self.actions = list(range(ACTION_NUM))

        self.trainer = DeepQNetworkTrainer(file_name=file_name)
        #self.path = self.trainer.logger.path_of(self.trainer.file_name)
        self.path = "/home/meip-users/tb3_op/deep_q_network/dqn_agent.h5"
        self.agent = DeepQNetworkAgent(1.0, self.actions)

        #ここから追加した変数 
        self.buffer_size = 1024

        #現在何エピソード目か
        self.episode_count = 0

        #stepに対してobserveごとに確認している。frameの仕組みがわかればなんとかなりそう。基本的に描画周りなのであまり使わなそう。
        self.frames = []

        #エピソード完了フラグ
        self.done = False

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
        self.goal_percentage = 0.

    def timer_callback(self):
        global robot_state_flag, sim_state_flag, all_steps, learning_stop, count_log, goal_log, goal_percentage_log
        print("all steps:" , all_steps)

        #TODO:繰り返し回数の設定
        if all_steps > 10000:
            if learning_stop == False:
                self.agent.model.save(self.path)
                reward_csv_path = "/home/meip-users/tb3_op/data/result.csv"

                with open(reward_csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["episode", "Reward", "step_count", "goal_log", "goal_percentage_log"])
                    for i in range(len(count_log)):
                        writer.writerow([i, self.trainer.reward_log[i], count_log[i], goal_log[i], goal_percentage_log[i] ])
                learning_stop = True
            print("learning done, goal count : ", self.goal_count)

        else:

            if not learning_stop:
                #self.get_logger().info('runnningggggg')
                if sim_state_flag == WAITSTOP and robot_state_flag == WAITRESET:
                    self.done = True

                if robot_state_flag == WAITSTOP and sim_state_flag == WAITSTOP:                    
                    #既存コードのtrain_loopに対応。
                    all_steps += 1
                    if all_steps % 100 == 0:
                        self.get_logger().info('now : "%d" steps' % all_steps)

                    #TODO:適切な位置に配置する
                    last_tf = current_tf

                    if not self.done:
                        best_action = self.best_action_learning()
                        action_log[best_action] += 1
                        send_msg = Int32()
                        #print("ba:", best_action)
                        #print("batype:", type(best_action))
                        send_msg.data = int(best_action)
                        self.publisher_.publish(send_msg)
                        self.get_logger().info('Publishing: "%d"' % send_msg.data)

                        #step_countの更新はbest_action_learning関数内で行う

                if self.done:
                    #停止命令を送信
                    send_msg = Int32()
                    #print("ba:", best_action)
                    #print("batype:", type(best_action))
                    send_msg.data = 99
                    self.publisher_.publish(send_msg)
                    self.get_logger().info('Publishing: "%d"' % send_msg.data)

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

        self.done = False

        if self.step_count > 200:
            print("too long episode detected")
            current_tf[0] = 0.2
            current_tf[1] = 0.2
            #reward -= 50
            self.done = True


        # collision : 0.1m以下とする
        if np.min(np_dis_list[np.nonzero(np_dis_list)]) < 0.1:
            print("!collision!")
            reward -= 30 * (1. - 1./(self.goal_count // 10 + 2))
            self.done = True
        
        #ゴール判定を初期は広げるというアプローチ。これ自体は悪くないがやや目的とずれる気もするので保留
        #goal_rad = 0.03 + 10. / (goal_count + 10.)
        goal_rad = 3.0
        if np.linalg.norm(target_vector) < goal_rad :
            print("!goal!")
            reward += 500
            self.done = True
            self.goal_count += 1
            is_goal = True

        return state, reward, self.done

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
        # TODO : 実装できたら解除する
        self.trainer.experiences.append(e)
        #self.trainer.current_experiences.append(e)

        #ある程度データが溜まったらそれ以降学習開始
        if not self.trainer.training and len(self.trainer.experiences) >= 128:
            #モデルを設定するなどなんやかんややっている。logger周り以外はなんとかなっていそうだが...
            self.trainer.begin_train(self.episode_count, self.agent)
            self.trainer.training = True
            print("training start !!!!!!!!!")          

        #ここでbatchを用いた学習を行う
        #多分大丈夫だと思うけどなあ。。。
        self.trainer.step(self.episode_count, self.step_count, self.agent, e)
        self.step_count += 1

        #ここは実装済み
        best_action = self.agent.policy(n_state)
        last_state = n_state
        last_action = best_action

        return best_action
    
    #複数回呼ばれる可能性がある関数なので注意。環境の初期化までは待機する必要がある
    def do_ifdone(self):
        global robot_state_flag, sim_state_flag, count_log, is_goal, goal_percentage_log

        #基本的に制御側(simulator)はwaitstartしない(動かすならwaitstop、止めるならwaitreset)
        #   待機　　　　　　開始(U-R側)　　停止(S側)  リセット(U-R側)   再開(S側)
        # S WAITSTOP  ->    ->    -> WAITRESET ->    ->     -> WAITSTOP ...
        # R WAITSTART -> WAITSTOP ->    ->     -> WAITSTART ->    ->    ...

        # S : waitstart & R : waitstart => 例外
        # S : waitstart & R : waitstop => 例外
        # S : waitstart & R : waitreset => 例外
        # S : waitstop & R : waitstart => ロボットはメッセージだけ投げる、Unity側で開始ボタンが押されたら学習等開始
        # S : waitstop & R : waitstop => doneじゃないなら続行、doneならロボットを止めてエピソード終了処理
        # S : waitstop & R : waitreset => 例外
        # S : waitreset & R : waitstart => 環境のリセットが完了したので再開
        # S : waitreset & R : waitstop => 環境の終了処理を待機中
        # S : waitreset & R : waitreset => 環境の終了処理を待機中
        

        
        #制御が動いておらず、環境のロボットがリセットされているなら再開
        if sim_state_flag == WAITRESET and robot_state_flag == WAITSTART:
            self.get_logger().info('Control restart. sim_state: "%d"' % sim_state_flag)
            print("goal_count", self.goal_count)
            count_log.append(self.step_count)
            goal_log.append(self.goal_count)
            if (len(goal_log) > 20):
                self.goal_percentage = (goal_log[-1] - goal_log[-11]) / 10.
                print("goal_percentage", self.goal_percentage)
            goal_percentage_log.append(self.goal_percentage)

            #エピソード数を加算
            self.done = False
            self.step_count = 0
            self.episode_count += 1
            is_goal = False
            self.trainer.episode_begin(self.episode_count, self.agent)
            sim_state_flag = WAITSTOP
        
        
        #制御は停止しておりロボットがリセットされていない場合は待機する
        elif sim_state_flag == WAITRESET:
            self.get_logger().info('Unity stopping episode. Waiting... sim_state: "%d"' % sim_state_flag)

            
        #ロボットが動いていてシミュレーターも動いているなら、まずシミュレータを待機状態にする。またエピソード終了の処理を実行
        elif sim_state_flag == WAITSTOP and (robot_state_flag == WAITSTOP or robot_state_flag == WAITRESET):
            sim_state_flag = WAITRESET
            self.get_logger().info('Control stopped. sim_state: "%d"' % sim_state_flag)

            #agent._teacher_modelを使っていることに注意。あとloggerも使っている。それ以外は不安要素はなさそう
            que_length = len(self.trainer.current_experiences)

            '''
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
            '''

            #experiencesにキューを追加
            #positive_experiences, negative_experiencesはそれぞれ保存されているキュー

            '''
            for i in range(1024):
                #終了時点のデータを忘れないようにする
                if len(copy_end_deque) != 0:
                    self.trainer.experiences.append(copy_end_deque.pop())
                else:
                    if len(copy_p_deque) == 0:
                        if len(copy_n_deque) != 0:
                            self.trainer.experiences.append(copy_n_deque.pop())
                    else:
                        self.trainer.experiences.append(copy_p_deque.pop())
            '''

            print("all:", len(self.trainer.experiences))
            print("end:", len(self.trainer.end_experiences))
            print("p:", len(self.trainer.positive_experiences))
            print("n:", len(self.trainer.negative_experiences))

            self.trainer.episode_end(self.episode_count, self.step_count, self.agent, count_log, is_goal, self.goal_count)



def main(args=None):

    rclpy.init(args=args)
    exec = SingleThreadedExecutor()
    action_publisher = ActionPublisher()
    state_publisher = SimStatePublisher()
    state_subscriber = RobotStateSubscriber()
    distance_subscriber =  DistanceSubscriber()
    position_subscriber = PositionSubscriber()
    tf_subscriber =  TfSubscriber()
    exec.add_node(action_publisher)
    exec.add_node(state_publisher)
    exec.add_node(state_subscriber)
    exec.add_node(distance_subscriber)
    exec.add_node(position_subscriber)
    exec.add_node(tf_subscriber)
    exec.spin()
    #rclpy.spin(action_publisher)
    #rclpy.spin(state_subscriber)
    exec.destroy_node()
    exec.shutdown()


if __name__ == '__main__':
    main()