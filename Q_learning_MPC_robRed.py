# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:05:40 2023

@author: zplrj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import random
states = ['NB', 'NM', 'NS', 'Z', 'PS', 'PM', 'PB']
#actions = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25] # Q learning action: RPM changing rate
actions = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1] # Q learning action: RPM changing rate
delta_e = 0.02


#Create a 2D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
q_values = np.array(pd.read_csv('C:/Users/zplrj/OneDrive/桌面/Research/PERL/Q_values_Green_5.csv'))

# Create a numpy array to hold the rewards for each state.
# The array is initialized to -1.
rewards = np.full(len(states), -1)
rewards[3] = 95.  # set the reward for the zero error to 100

# input membership function: error
err_interval = 0.01
aaa = 10 # nmumber of intervals in one side of the trangle
err_axis = np.arange(-6*aaa*err_interval,6*aaa*err_interval+err_interval,err_interval)
k1 = 1/(-2*aaa*err_interval)
k2 = 1/(2*aaa*err_interval)


#define a function that determines if the specified error is a terminal state
def is_terminal_state(current_error):
  if np.abs(current_error) >= delta_e:
    return False
  else:
    return True

# fuzzy current error
def error_fuzzy (current_error):
    if current_error<=-6*aaa*err_interval:
        return [[0,1]]
    elif current_error>-6*aaa*err_interval and current_error<=-4*aaa*err_interval:
        pNB = k1 * current_error - 2
        pNM = k2 * current_error + 3
        return [[0, pNB],[1,pNM]]
    elif current_error > -4 * aaa * err_interval and current_error <= -2 * aaa * err_interval:
        pNM = k1 * current_error - 1
        pNS = k2 * current_error + 2
        return [[1, pNM],[2,pNS]]
    elif current_error > -2 * aaa * err_interval and current_error <= 0 * aaa * err_interval:
        pNS = k1 * current_error
        pZ = k2 * current_error + 1
        return [[2, pNS],[3,pZ]]
    elif current_error > 0 * aaa * err_interval and current_error <= 2 * aaa * err_interval:
        pZ = k1 * current_error +1
        pPS = k2 * current_error
        return [[3, pZ],[4,pPS]]
    elif current_error > 2 * aaa * err_interval and current_error <= 4 * aaa * err_interval:
        pPS = k1 * current_error +2
        pPM = k2 * current_error - 1
        return [[4, pPS],[5,pPM]]
    elif current_error > 4 * aaa * err_interval and current_error <= 6 * aaa * err_interval:
        pPM = k1 * current_error +3
        pPB = k2 * current_error - 2
        return [[5, pPM],[6,pPB]]
    else:
        return [[6, 1]]


#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_error, epsilon,states):
  #if a randomly chosen value between 0 and 1 is less than epsilon,
  #then choose the most promising value from the Q-table for this state.
  temp = error_fuzzy(current_error)
  if len(temp) == 1:
      # fuzzy_state = temp[0][0]
      fuzzy_state_index = temp[0][0]
      if np.random.random() < epsilon:
          return [[np.argmax(q_values[fuzzy_state_index,]),1]]
      else:  # choose a random action
          return [[np.random.randint(len(actions)),1]]
  else:
      # fuzzy_state1 = temp[0][0]
      fuzzy_state_index1 = temp[0][0]
      fuzzy_state_p1 = temp[0][1]
      # fuzzy_state2 = temp[1][0]
      fuzzy_state_index2 = temp[1][0]
      fuzzy_state_p2 = temp[1][1]
      if np.random.random() < epsilon:
            temp_ind_1 = np.argwhere(q_values[fuzzy_state_index1,]==np.amax(q_values[fuzzy_state_index1,])).reshape(-1)
            action_index_1 = temp_ind_1[np.random.randint(len(temp_ind_1))]
            temp_ind_2 = np.argwhere(q_values[fuzzy_state_index2,]==np.amax(q_values[fuzzy_state_index2,])).reshape(-1)
            action_index_2 = temp_ind_2[np.random.randint(len(temp_ind_2))]
            return [[action_index_1,fuzzy_state_p1],[action_index_2,fuzzy_state_p2]]
            #return [[np.argmax(q_values[fuzzy_state_index1,]),fuzzy_state_p1],[np.argmax(q_values[fuzzy_state_index2,]),fuzzy_state_p2]]
      else:  # choose a random action
            return [[np.random.randint(len(actions)),fuzzy_state_p1], [np.random.randint(len(actions)),fuzzy_state_p2]]


#define a function that will get the next parameter change rate
def get_next_parameter_change_rate(next_action):
  if len(next_action) == 1:
      next_action_index = next_action[0][0]
      an_array = np.array(actions[next_action_index])
      return an_array
  else:
      next_action_index1 = next_action[0][0]
      next_action_index2 = next_action[1][0]
      next_action_p1 = next_action[0][1]
      next_action_p2 = next_action[1][1]
      an_array1 = np.array(actions[next_action_index1])
      multiplied_array1 = an_array1 * next_action_p1
      an_array2 = np.array(actions[next_action_index2])
      multiplied_array2 = an_array2 * next_action_p2
      return np.add(multiplied_array1,multiplied_array2)

# define training parameters
epsilon = 1  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.9  # the rate at which the AI agent should learn

# train the agent
# use the algorithm speed to train the agent, the speed profile is repeated for 10 times
#vs_target_raw = np.array(pd.read_csv('/Users/zhangpeng/Desktop/experiment_2.csv',header=None))
#vs_target = vs_target_raw['m_speed_21']
#vs_target_raw = pd.read_csv('/Users/zhangpeng/Desktop/experiment_2.csv')
#vs_target = vs_target_raw['m_speed_21']
data_delta_t = 0.2 # 0.2 s
#%%
from _thread import *
#from _thread import start_new_thread
import socket
import sys
import time
import pandas as pd
import numpy as np
##from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
import re
##忽略Broken Pipe
##signal(SIGPIPE, SIG_IGN)
##读取rob1数据并且转换为lookup table
experiment = pd.read_csv('C:/Users/zplrj/OneDrive/桌面/Research/PERL/experiment_2.csv')
speed_3 = experiment['m_speed_19']
speed_4 = experiment['m_speed_4']
real_speed = pd.read_csv('C:/Users/zplrj/OneDrive/桌面/Research/PERL/real_speed.csv')
real_speed['Y'] = real_speed['Y'].round(decimals=3)
real_speed['X'] = real_speed['X'].round(decimals=0)
RPM = real_speed['X']
should_speed = real_speed['Y']
speed_17 = experiment['m_speed_34']
closest_list_1 = []
##for i in range(0,len(speed_17)):
for i in range(0,110):    
    given_value = speed_17[i]
    a_list = should_speed
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(a_list, key=absolute_difference_function)
    closest_list_1.append(closest_value)
RPM_send_1 = []
for i in range(0,len(closest_list_1)):
    index = real_speed.index[real_speed['Y'] == closest_list_1[i]].tolist()
    RPM_send_1.append(RPM[index[0]])


##rob1的实验位移
distance_exp_1 = []

for i in range(0,len(closest_list_1)):
    #t_distance = (time_exp_2.loc[i]*closest_list_2[i])
    t_distance = (0.4*closest_list_1[i])
    distance_exp_1.append(t_distance)
df_distance_exp_t_1 = pd.DataFrame(distance_exp_1)
##计算rob1实验位移
distance_exp_1 = []
distance_exp_1 = df_distance_exp_t_1.cumsum()
#temp_1 = []
#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective(v, dt, w1, w2, w3, x_tar, v_tar, x0):
    f1 = np.zeros(len(v) - 1)
    f2 = np.zeros(len(v) - 1)
    f3 = np.zeros(len(v) - 1)
    for i in range(1, len(v)):
        f1[i-1] = (sum(dt * v[1:i]) + x0 - x_tar[i]) ** 2
        f2[i-1] = (v[i] - v_tar[i]) ** 2
        f3[i-1] = ((v[i] - v[i - 1]) / dt) ** 2

    f = w1 * sum(f1) + w2 * sum(f2) + w3 * sum(f3)
    return f


def constraint1(v, idx, a_min):
    c1 = (v[idx] - v[idx - 1]) / dt - a_min
    return c1


def constraint2(v, idx, a_max):
    c2 = a_max - (v[idx] - v[idx - 1]) / dt
    return c2


# optimize
dt = 0.1 # data point time interval

T = 10  # predict period 1 s
w1 = 1.5 # location weight in the obj
w2 = 0.7 # speed weight in the obj
w3 = 0.7 # acceleration rate in the obj
#w1 = 4 # location weight in the obj
#w2 = 0.1 # speed weight in the obj
#w3 = 0.5 # acceleration rate in the obj

a_max = 2
a_min = -1.5
#%%
##读取rob2数据并且转换为lookup table
#real_speed = pd.read_csv('C:/Users/zplrj/OneDrive/桌面/Research/PERL/real_speed.csv')
real_speed = pd.read_csv('C:/Users/zplrj/OneDrive/桌面/Research/PERL/interpolated_rpm_robot_5_80.csv')

real_speed['Y'] = real_speed['Y'].round(decimals=3)
real_speed['X'] = real_speed['X'].round(decimals=0)
RPM = real_speed['X']
should_speed = real_speed['Y']
speed_18 = experiment['m_speed_22']
closest_list_2 = []
##for i in range(0,len(speed_17)):
for i in range(0,28):    
    given_value = speed_18[i]
    a_list = should_speed
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(a_list, key=absolute_difference_function)
    closest_list_2.append(closest_value)
RPM_send_2 = []
for i in range(0,len(closest_list_2)):
    index = real_speed.index[real_speed['Y'] == closest_list_2[i]].tolist()
    RPM_send_2.append(RPM[index[0]])
temp_1 = []
temp_2 = []
_ThreadCount = 0
raw = ""
a = "AA300B"
rob_speed = 0
W_fin = 0
delay = 0.1
#%%
run_threads = True
#Q_learning
def clientthread_0(conn):
    global run_threads
    global raw
    global a
    global temp
    global W_fin
    global rob_speed
    global given_value
    global _ThreadCount
    RPM = 300
    
    while run_threads:
        for i in range(0,len(RPM_send_1)):            
            time.sleep(0.4)
            #time.sleep(0.2)
            _ThreadCount += 1
            error = given_value - rob_speed # calculate speed error
            error = ((1 / (1 + np.exp(-12*error)))-0.5)*(6*aaa*err_interval/0.5) # condition in range [-6*aaa*err_interval,6*aaa*err_interval]
       
            old_RPM = RPM # save the old RPM
            old_error = error # save the old speed error
            next_action = get_next_action(old_error, epsilon, states) # get the next action based on old_error
            change_rate = get_next_parameter_change_rate(next_action) # get the next RMP charging rate
            RPM = old_RPM + old_RPM * change_rate # update the RPM
            #print("RPM = " + str(RPM))  
            ################## assign robot the new RPM
            a = str('AA') + str(int(RPM)) + str('B')
            #print(a)
            #print('ThreadCount is %s    ' % _ThreadCount)
        a = 'AA000000B'
        rob_speed = 0
        break  # Exit the while loop after setting a to 'AA000B'
#MPC            
def clientthread_1(conn):
    global run_threads
    global raw
    global a
    global temp
    global W_fin
    global rob_speed
    global given_value
    global _ThreadCount
    while run_threads:
        
        count = _ThreadCount
        T1 = time.perf_counter()


        #print ('rob_speed:%s(m/s)'%(rob_speed))
        #print ('rob_time:%s(s)'%(rob_t))
        #print ('rob_time:%s(count)'%(count))
        #print ('distance:%s(m)'%(W_fin))
        
        
        
        # input current speed and position, replaced by observed speed and position
        v_now = rob_speed
        rob_t1 = count
        x_now = W_fin

        # input target speeds and positions, may replaced with a set of speeds and a set of positions
        #x_target = np.zeros(T)
        list_distance_exp_1 = distance_exp_1[0].to_list()
        x_target = list_distance_exp_1[rob_t1:]
        #v_target = 0.6 * np.ones(T)
        v_target = closest_list_1[rob_t1:]


        # bounds for decision variables, speed bound
        b = (0, 0.5)
        bnds = []
        for i in range(T):
            bnds.append(b)

        # constraints
        con1 = {'type': 'eq', 'fun': lambda v: v[0] - v_now}  # initial speed
        # con2 = {'type': 'ineq', 'fun': constraint1, 'args': a_min}
        # con3 = {'type': 'ineq', 'fun': constraint2, 'args': a_max}
        cons = [con1]
        for ii in range(1, T):
            cons.append({'type': 'ineq', 'fun': constraint1, 'args': (ii, a_min)})
            cons.append({'type': 'ineq', 'fun': constraint2, 'args': (ii, a_max)})

        x0 = 0 * np.ones(T)
        solution = minimize(objective, x0, args=(dt, w1, w2, w3, x_target, v_target, x_now), method='SLSQP', bounds=bnds,
                            constraints=cons, options={'disp': False})

        v_plan = solution.x
        given_value = v_plan[9]
        T2 = time.perf_counter()

        #print('运行时间:%s秒'%(T2-T1))
        
        
def clientthread_2(conn):
    global run_threads
    global a
    
    while run_threads:
        conn.send(bytes(a,"utf-8"))
        print('speed sent = %s'%a)
        time.sleep(delay)   
    

    
def clientthread_3(conn):
    global run_threads
    global raw 
    global W_fin
    global rob_speed
    buffer = ""
    
    while run_threads:
        buffer = conn.recv(1024)
        raw = buffer.decode("utf-8")
        #print(raw)
        temp.append(raw)
        temp_t=raw.split(',')
        temp_final_1 = []
        for element in temp_t:
            if re.match("^2;.*;.*",element):
                temp_final_1.append(element)
        temp_final1=[]        
        for i in range(0,len(temp_final_1)):
            temp_t=temp_final_1[i].split(';')
            for j in range(0,len(temp_t)):
                temp_final1.append(temp_t[j])
        temp_final11=[]
        temp_final12=[]
        for i in range(0,len(temp_final1)):
            if i%3==0:
                temp_final12.append(temp_final1[i])
            else:
                temp_final11.append(temp_final1[i])
        time_raw_1 = []
        distance_raw_1 = []
        for i in range(0,len(temp_final11)):
            if i%2==0:
                time_raw_1.append(temp_final11[i])
            else:
                distance_raw_1.append(temp_final11[i])
        time_raw_clean_1 = []
        distance_raw_clean_1 = []
        for i in range(0, len(time_raw_1)-1):
            if i % 1 == 0:
                try:
                    distance_raw_clean_1.append(float(distance_raw_1[i]))
                    time_raw_clean_1.append(float(time_raw_1[i]))
                except ValueError:
                    print ("error on line",i)
        
        E = np.sum(time_raw_clean_1)
        
        
        distance_real_1 = []
        for i in range(0, len(time_raw_clean_1)-1):
            distance_temp = distance_raw_clean_1[i]*time_raw_clean_1[i]
            distance_real_1.append(distance_temp)
            

        #print(len(distance_real_1))
        W = np.sum(distance_real_1)
        distance_real_1_fin.append(W)
        W_fin = round((np.sum(distance_real_1_fin))/1000,3) 
        
        rob_time.append(E)
        rob_t = int((np.sum(rob_time))/2500)
        rob_speed = round(np.mean(distance_raw_clean_1),3)
        
#%%
host = '192.168.1.103'
port = 56791
tot_socket = 5
list_sock = []
temp = []
rob_time = []
distance_real_1_fin = []
#count = 0
for i in range(0,tot_socket):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((host, port+i))
    s.listen(5)
    list_sock.append(s)            
    print ("Server listening on" + str(host)+ ' port '+ str(port+i))
#%%

for s in list_sock:
    s.settimeout(1.0)  # Set timeout for accept

try:
    while run_threads:
        for j in range(len(list_sock)):
            if j == 0:
                try:
                    conn, addr = list_sock[j].accept()
                    print('first thread')
                    print('Connected with ' + addr[0] + ':' + str(addr[1]))
                    start_new_thread(clientthread_2, (conn,))  # send speed sequence
                    start_new_thread(clientthread_3, (conn,))  # receive and process data
                    #time.sleep(0.5)
                    start_new_thread(clientthread_0, (conn,))  # count time
                    start_new_thread(clientthread_1, (conn,))  # run MPC
                except socket.timeout:
                    continue  # Continue checking if run_threads is still True
except KeyboardInterrupt:
    run_threads = False

# Close sockets
for s in list_sock:
    s.close()

input("Press Enter to stop the server...")
run_threads = False



#%%
##Separate the data by ','
temp_f=[]
temp_test = []
for i in range(0,len(temp)):
    temp_t=temp[i].split(',')
    for j in range(0,len(temp_t)-1):
        temp_f.append(temp_t[j])
#%%
##Find the data with specific label
import re
temp_final_1 = []
for element in temp_f:
    if re.match("^2;.*;.*",element):
        temp_final_1.append(element)
temp_final_1
#%%
##Separate data from rob1
temp_final1=[]
temp_test = []
for i in range(0,len(temp_final_1)):
    temp_t=temp_final_1[i].split(';')
    for j in range(0,len(temp_t)):
        temp_final1.append(temp_t[j])
#%%
##filter the data
temp_final11=[]
temp_final12=[]
for i in range(0,len(temp_final1)):
    if i%3==0:
        temp_final12.append(temp_final1[i])
    else:
        temp_final11.append(temp_final1[i])
#%%
##get the Temporary speed of rob1
time_raw_1 = []
distance_raw_1 = []
for i in range(0,len(temp_final11)):
    if i%2==0:
        time_raw_1.append(temp_final11[i])
    else:
        distance_raw_1.append(temp_final11[i])
#%%
##Filter the wrong data
time_raw_clean_1 = []
distance_raw_clean_1 = []
for i in range(0, len(time_raw_1)-1):
    if i % 1 == 0:
        try:
            distance_raw_clean_1.append(float(distance_raw_1[i]))
            time_raw_clean_1.append(float(time_raw_1[i]))
        except ValueError:
            print ("error on line",i)
#%%
##Filter again and combine the data
time_1 = []
distance_1 = []
for i in range(0,len(time_raw_clean_1)):
    if time_raw_clean_1[i]>=0 and distance_raw_clean_1[i]>0:
        time_1.append(time_raw_clean_1[i])
        distance_1.append(distance_raw_clean_1[i])
#%%
df_time_clean_1 = pd.DataFrame(time_1)
df_distance_clean_1 = pd.DataFrame(distance_1)
#%%
##get the distance of rob1
distance_real_t_1=[]
for i in range(0,len(time_1)):
    t_distance = distance_1[i]*time_1[i]/1000
    distance_real_t_1.append(t_distance)
df_distance_real_t_1 = pd.DataFrame(distance_real_t_1)
#%%
##get the rob1's real distance
distance_real_1 = []
distance_real_1 = df_distance_real_t_1.cumsum() 
#%%
##calculate the time of rob1 use
time_real_t_1=[]
for i in range(0,len(time_1)):
    t_time = time_1[i]/1000
    time_real_t_1.append(t_time)
df_time_real_t_1 = pd.DataFrame(time_real_t_1)

time_real_1 = []
time_real_1 = df_time_real_t_1.cumsum() 
#%%
##use the last value of time_real_1 to calculate time_exp_1
time_exp_t_1 = []
time_exp_1 = []
for i in range(0,110):
    t_time = 0.4
    time_exp_t_1.append(t_time)
df_time_exp_1 = pd.DataFrame(time_exp_t_1)
time_exp_1 = df_time_exp_1.cumsum() 
#%%
##rob1's distance
distance_exp_1 = []

for i in range(0,len(time_exp_1)-2):
    #t_distance = (time_exp_1.loc[i]*closest_list_1[i])
    t_distance = (0.4*closest_list_1[i])
    distance_exp_1.append(t_distance)
df_distance_exp_t_1 = pd.DataFrame(distance_exp_1)
##rob1's experimental distance
distance_exp_1 = []
distance_exp_1 = df_distance_exp_t_1.cumsum() 
#%%
##rob1's average speed 
import statistics
speed_ave_1 = []
i_list_1 = []
speed_sum_1 = 0
for i in range(0, len(time_1)):
    if i % 118 == 0:
        i_list_1.append(i)
        
        
for i in range(0, len(i_list_1)-1):
    average_speed = statistics.mean(distance_1[i_list_1[i]:i_list_1[i+1]])
    speed_ave_1.append(average_speed)
#%%
##shrink the data from rob1
time_real_skew_1 = []
distance_real_skew_1 = []
for i in range(0, len(time_real_1)):
    if i % 118 == 0:
        time_real_skew_1.append(time_real_1.loc[i])
        distance_real_skew_1.append(distance_real_1.loc[i])    
df_time_real_skew_1 = pd.DataFrame(time_real_skew_1)
df_distance_real_skew_1 = pd.DataFrame(distance_real_skew_1)
#%%
time_real_skew_1 = []
distance_real_skew_1 = []
cumulative_distance = 0
time_interval = 0.4
for i in range(0, len(time_real_1)):
    if i % 118 == 0:
        time_real_skew_1.append(time_real_1.loc[i])
        # distance_real_skew_1.append(distance_real_1.loc[i])
for speed in speed_ave_1:
    cumulative_distance += speed * time_interval
    distance_real_skew_1.append(cumulative_distance)
    
df_time_real_skew_1 = pd.DataFrame(time_real_skew_1)
df_distance_real_skew_1 = pd.DataFrame(distance_real_skew_1)
#%%
##rob1's experimental speed
speed_exp_1 = []
for i in range(0,110):
    t_distance = closest_list_1[i]
    speed_exp_1.append(t_distance)
#%%
##print rob1's distance
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
ax.plot
ax.plot(distance_real_skew_1[0:110],'r.',label="RobotRed")
ax.plot(distance_exp_1[0:110],'b',label="Experiment ")
#ax.plot(x,app,label="robotRed")
plt.xlabel('time index')
plt.ylabel('distance(m)')
ax.legend()
plt.rc('font',size=18)
#%%
distance_difference_real = []
distance_difference_exp = []
for i in range(0,45):    
    diff_exp = distance_real_skew_1[i]-distance_exp_1.loc[i]
    distance_difference_exp.append(diff_exp)
#%%
##print rob1's speed
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(speed_ave_1[0:110],'r.',label='robotRed')
ax.plot(closest_list_1[0:110],'b',label='Experiment')
ax.plot
ax.legend()
plt.xlabel('time index')
plt.ylabel('speed(m/s)')
plt.rc('font',size=18)
#%%
##calculate rob1's speed RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(speed_exp_1[40:80],speed_ave_1[40:80])
rmse = sqrt(mse)
print(rmse)
#%%
##calculate rob1's location RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(distance_exp_1[40:80],distance_real_skew_1[40:80])
rmse = sqrt(mse)
print(rmse)
#%%
distance_exp_1_list = distance_exp_1[0].values.tolist()
df_distance_real_skew_1_list = df_distance_real_skew_1[0].values.tolist()
dict_print = {'speed_exp_1':speed_exp_1[0:120],'speed_ave_1':speed_ave_1[0:120],'distance_exp_1':distance_exp_1_list[0:120],'distance_real_skew_1':df_distance_real_skew_1_list[0:120]}