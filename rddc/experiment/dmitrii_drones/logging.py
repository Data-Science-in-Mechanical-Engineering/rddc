#!/usr/bin/env python3

import rospy
from crazyswarm.ros_ws.src.crazyswarm.scripts.pycrazyswarm import GenericLogData
import numpy as np

class bufferStateLogger():
    def __init__(self, crazyswarm_launch_f_path='/home/franka_panda/crazyswam/crazyswarm/ros_ws/src/crazyswarm/launch/hover_swarm.launch',\
                 buffer_size=10) -> None:
        self.launch_f = crazyswarm_launch_f_path
        self.buffer_size = buffer_size
        self.buffer = list()
        # self.topicName = 'log1'

    # def __read_launch_file(self):
    #     self.data_types = ['time_stamp']
    #     self.n_data_types = None

    #     data_line = None
    #     with open(self.launch_f, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             found = line.find('genericLogTopic_'+ self.topicName+'_Variables')
    #             if found > -1:
    #                 data_line = line
    #     #print(data_line)
    #     while True:
    #         i_start = data_line.find('"') + 1
    #         if i_start == 0:
    #             break
    #         i_end = data_line.find('"', i_start)
    #         self.data_types.append(data_line[i_start : i_end])
    #         data_line = data_line[i_end+1 : ]

    def __data_callback(self, data):
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs
        
        time = float(__t_sec) + 1e-9 * __t_nsec
        __data = [time]

        for v in data.values:
            __data.append(v)

        print("Logged data:\n\t{}\n".format(__data))
        self.buffer.append(__data)
        if len(self.buffer)>self.buffer_size:
            del(self.buffer[0])

    def log_run(self):
        # self.__read_launch_file()
        rospy.init_node('Logger', anonymous=True)
        rospy.Subscriber('/cf4/log1', GenericLogData, self.__data_callback)
        rospy.spin()

    def __del__(self):
        self.csv_f.close()
        print("destructor")

    def retrieve_state(self):
        state = np.array(12)
        # +1 since the first element in __data is timestamp
        for idx in range(9):
            state[idx] = self.buffer[-1][idx+1]
        for idx in [9, 10, 11]:
            state[idx] = (self.buffer[-1][idx-2] - self.buffer[-2][idx-2])/(self.buffer[-1][0] - self.buffer[-2][0])
        return state

    def retrieve_time(self):
        return self.buffer[-1][0]


if __name__ == '__main__':
    x = bufferStateLogger()
    x.log_run()
