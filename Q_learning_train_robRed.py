#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import random
states = ['NB', 'NM', 'NS', 'Z', 'PS', 'PM', 'PB']
#actions = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25] # Q learning action: RPM changing rate
actions = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1] # Q learning action: RPM changing rate

delta_e = 0.02


#Create a 2D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
#q_values = np.zeros((len(states), len(actions)))
q_values = np.array(pd.read_csv('/Users/zhangpeng/Desktop/Q_values_robRed1.csv'))
# Create a numpy array to hold the rewards for each state.
# The array is initialized to -1.
#rewards = np.full(len(states), -1)
#rewards[3] = 95.  # set the reward for the zero error to 100
rewards = [-1000,-100,-10,100,-10,-100,-1000]

# input membership function: error
#err_interval = 0.01
err_interval = 0.005
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
  temp_error = error_fuzzy(current_error)
  if len(temp_error) == 1:
      # fuzzy_state = temp[0][0]
      fuzzy_state_index = temp_error[0][0]
      if np.random.random() < epsilon:
          return [[np.argmax(q_values[fuzzy_state_index,]),1]]
      else:  # choose a random action
          return [[np.random.randint(len(actions)),1]]
  else:
      # fuzzy_state1 = temp[0][0]
      fuzzy_state_index1 = temp_error[0][0]
      fuzzy_state_p1 = temp_error[0][1]
      # fuzzy_state2 = temp[1][0]
      fuzzy_state_index2 = temp_error[1][0]
      fuzzy_state_p2 = temp_error[1][1]
      
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
epsilon = 0.9  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.9  # the rate at which the AI agent should learn

# train the agent
# use the algorithm speed to train the agent, the speed profile is repeated for 10 times
#vs_target_raw = np.array(pd.read_csv('/Users/zhangpeng/Desktop/experiment_2.csv',header=None))
#vs_target = vs_target_raw['m_speed_21']
vs_target_raw = pd.read_csv('/Users/zhangpeng/Desktop/experiment_2.csv')
vs_target = vs_target_raw['m_speed_21']
data_delta_t = 0.2 # 0.2 s
#%%
from _thread import *
import socket
import sys
import time
import pandas as pd
import numpy as np
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
import re
##ignore Broken Pipe
signal(SIGPIPE, SIG_IGN)
##read rob1's speed sequience and transfer to lookup table
experiment = pd.read_csv('/Users/zhangpeng/Desktop/experiment_2.csv')
speed_3 = experiment['m_speed_19']
speed_4 = experiment['m_speed_4']
real_speed = pd.read_csv('/Users/zhangpeng/Desktop/real_speed.csv')
real_speed['Y'] = real_speed['Y'].round(decimals=3)
real_speed['X'] = real_speed['X'].round(decimals=0)
RPM = real_speed['X']
should_speed = real_speed['Y']
speed_17 = experiment['m_speed_60']
#speed_17 = experiment['m_speed_26']
closest_list_1 = []
##for i in range(0,len(speed_17)):
for i in range(0,55):    
    given_value = speed_17[i]
    a_list = should_speed
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(a_list, key=absolute_difference_function)
    closest_list_1.append(closest_value)
RPM_send_1 = []
for i in range(0,len(closest_list_1)):
    index = real_speed.index[real_speed['Y'] == closest_list_1[i]].tolist()
    RPM_send_1.append(RPM[index[0]])



distance_exp_1 = []

for i in range(0,len(closest_list_1)):
    #t_distance = (time_exp_2.loc[i]*closest_list_2[i])
    t_distance = (0.4*closest_list_1[i])
    distance_exp_1.append(t_distance)
df_distance_exp_t_1 = pd.DataFrame(distance_exp_1)
##calculate rob1's experimental distance
distance_exp_1 = []
distance_exp_1 = df_distance_exp_t_1.cumsum()
#temp_1 = []
#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def clientthread_1(conn):
    buffer_1=""
    while True:

        #print('trying to receiving')
        data = conn.recv(1024)
        buffer_1 = data.decode("utf-8")
        temp_1.append(buffer_1)
        print ('data from rob1')
        print(buffer_1)
        
        
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
w1 = 0.8 # location weight in the obj
w2 = 1.4 # speed weight in the obj
w3 = 0.2 # acceleration rate in the obj

a_max = 2
a_min = -1.5
#%%
##read rob2's speed sequence and transfer to lookup table
real_speed = pd.read_csv('/Users/zhangpeng/Desktop/real_speed.csv')
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
a = "AA200B"
rob_speed = 0
W_fin = 0
delay = 0.1
#%%
def clientthread_0(conn):
    global _ThreadCount
    while True:
        for i in range(0,len(RPM_send_1)):            
            time.sleep(0.4)
            _ThreadCount += 1
            #print('ThreadCount is %s    ' % _ThreadCount)
            
            
def clientthread_1(conn):
    global raw
    global a
    global temp
    global W_fin
    global rob_speed
    T1 = time.perf_counter()
    # initial values shall be roughly set based on vs_target[0]
    RPM = 200
    count = 0
    ED = 100
    for t in range(0,1000):
        ################## assign robot the initial RPM
        #a = str('FF') + str(int(RPM)) + str('E')
        #print(a)
        ################## get real speed and error
        error = speed_17[t] - rob_speed # calculate speed error
        error = ((1 / (1 + np.exp(-12*error)))-0.5)*(6*aaa*err_interval/0.5) # condition in range [-6*aaa*err_interval,6*aaa*err_interval]
        ################## read current time
        cur_t = 0.1 #
        
        ################## read actual speed from the robot
        #v_robot_actual = vs_target[t] + random.uniform(0, 0.5)
        #count = count + 1

        while not (is_terminal_state(error) and ED < 50 and count > 10): ##and cur_t < (t+1) * data_delta_t: # loop until the speed error is less than the threshold or the current time exceeds
            old_RPM = RPM # save the old RPM
            old_error = error # save the old speed error
            next_action = get_next_action(old_error, epsilon, states) # get the next action based on old_error
            change_rate = get_next_parameter_change_rate(next_action) # get the next RMP charging rate
            RPM = old_RPM + old_RPM * change_rate # update the RPM
            
            ################## assign robot the new RPM
            a = str('AA') + str(int(RPM)) + str('B')
            time.sleep(1)
            #print(RPM)
            print(rob_speed)
            ################### read new speed from the robot
            #v_new_robot_actual = vs_target[t] + random.uniform(0, 0.2)


            error = speed_17[t] - rob_speed # calculate new speed error
            error = ((1 / (1 + np.exp(-12*error))) - 0.5) * ( 6 * aaa * err_interval / 0.5)  # condition the new error in range [-6*aaa*err_interval,6*aaa*err_interval]

            # fuzzy the old error and calculate the old q value
            fuzzy_old_e = error_fuzzy(old_error)
            if len(fuzzy_old_e) == 1:
                fuzzy_state_index = fuzzy_old_e[0][0]
                old_q_value = q_values[fuzzy_state_index,next_action[0][0]]
            else:
                fuzzy_state_index1 = fuzzy_old_e[0][0]
                fuzzy_state_p1 = fuzzy_old_e[0][1]
                fuzzy_state_index2 = fuzzy_old_e[1][0]
                fuzzy_state_p2 = fuzzy_old_e[1][1]
                old_q_value = fuzzy_state_p1 * q_values[fuzzy_state_index1,next_action[0][0]] + fuzzy_state_p2 * q_values[fuzzy_state_index2,next_action[1][0]]

            # fuzzy new error and calculate the reward
            fuzzy_new_e = error_fuzzy(error)
            if len(fuzzy_new_e) == 1:
                state_index = fuzzy_new_e[0][0]
                reward = rewards[state_index]
            else:
                state_index1 = fuzzy_new_e[0][0]
                state_index2 = fuzzy_new_e[1][0]
                state_p1 = fuzzy_new_e[0][1]
                state_p2 = fuzzy_new_e[1][1]
                reward = rewards[state_index1]*state_p1+rewards[state_index2]*state_p2

                V = state_p1 * np.max(q_values[state_index1,]) + state_p2 * np.max(q_values[state_index2,])
                temporal_difference = reward + (discount_factor * V) - old_q_value

                # update the Q-value for the previous state and action pair
                old_q_values = q_values
                q_values[fuzzy_state_index1, next_action[0][0]] = q_values[fuzzy_state_index1,next_action[0][0]] + (learning_rate * temporal_difference * fuzzy_state_p1)
                q_values[fuzzy_state_index2, next_action[1][0]] = q_values[fuzzy_state_index2, next_action[1][0]] + (learning_rate * temporal_difference * fuzzy_state_p2)
                distance = euclidean_distances(q_values,old_q_values)
                #ED = sum(sum(distance))
                ED = np.mean(distance)
                count = count + 1
                print(ED)
                #print(count)
    print('Complete!')
    T2 = time.perf_counter()

        
        
def clientthread_2(conn):
    global a
    while True:
        conn.send(bytes(a,"utf-8"))
        #print('speed sent = %s'%a)
        time.sleep(delay)   
    

    
def clientthread_3(conn):
    global raw 
    global W_fin
    global rob_speed
    buffer = ""
    while True:
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
        #print(rob_speed)
#%%
host = '192.168.1.103'
port = 56790
tot_socket = 5
list_sock = []
temp = []
rob_time = []
distance_real_1_fin = []
count = 0
for i in range(0,tot_socket):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((host, port+i))
    s.listen(5)
    list_sock.append(s)            
    print ("Server listening on" + str(host)+ ' port '+ str(port+i))
#%%
while 1:
    for j in range(len(list_sock)):
        if j == 0:
            conn, addr = list_sock[j].accept()
            print('first thread')
            print ('Connected with ' + addr[0] + ':' + str(addr[1]))
            start_new_thread(clientthread_2 ,(conn,))#send speed sequence
            #time.sleep(4)
            start_new_thread(clientthread_3 ,(conn,))#receive and process data
            #time.sleep(4.5)
            time.sleep(0.5)
            start_new_thread(clientthread_0 ,(conn,))#count time
            start_new_thread(clientthread_1 ,(conn,))#run MPC
        #if j == 1:
            #conn, addr = list_sock[j].accept()
            #print('second thread')
            #print ('Connected with ' + addr[0] + ':' + str(addr[1]))
            #start_new_thread(clientthread_2 ,(conn,))
s.close()

#%%
#q_values_list = np.array(q_values).tolist()
#dict = q_values_list[0:7]
df_q_values = pd.DataFrame(q_values)
#df_q_values.to_csv('Q_values_17.csv', index = False, header= False)
df_q_values.to_csv('Q_values_robRed1.csv', index = False)

#%%
import numpy as np

with open("Q_values_10.csv") as file_name:
    array_Q_values_A = np.loadtxt(file_name, delimiter=",")

with open("Q_values_13.csv") as file_name:
    array_Q_values_B = np.loadtxt(file_name, delimiter=",")
distance = euclidean_distances(array_Q_values_A,array_Q_values_B)
ED_A = np.mean(distance)
print(ED_A)
#%%

