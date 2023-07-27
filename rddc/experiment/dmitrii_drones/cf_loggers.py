#!/usr/bin/env python3

import rospy
# from crazyswarm.ros_ws.src.crazyswarm.scripts.pycrazyswarm import GenericLogData
# from crazyswarm.msg import GenericLogData
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np

class bufferStateLogger():
    def __init__(self, crazyswarm_launch_f_path='/home/franka_panda/crazyswam/crazyswarm/ros_ws/src/crazyswarm/launch/dmitrii_logging.launch',\
                 buffer_size=10) -> None:
        self.launch_f = crazyswarm_launch_f_path
        self.buffer_size = buffer_size
        self.pos_buffer = [[0.0, 0.0, 0.0, 0.0]]
        self.vel_buffer = [[0.0, 0.0, 0.0, 0.0]]
        self.angle_buffer = [[0.0, 0.0, 0.0, 0.0]]
        self.gyro_buffer = [[0.0, 0.0, 0.0, 0.0]]

    def __pos_callback(self, data):
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs
        
        time = float(__t_sec) + 1e-9 * __t_nsec
        __data = [time]

        for v in data.values:
            __data.append(v)

        if (time-self.pos_buffer[-1][0] < 1e-6):
            print(f"[logger] Spotted a repeated pos log")
            print(f"[logger] received data: {__data}")
            print(f"[logger] last data: {self.pos_buffer[-1]}")
            print(f"[logger] -> skipping the update")
            return

        # print("Logged pos:\n\t{}\n".format(__data))
        self.pos_buffer.append(__data)
        if len(self.pos_buffer)>self.buffer_size:
            del(self.pos_buffer[0])

    def __vel_callback(self, data):
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs
        
        time = float(__t_sec) + 1e-9 * __t_nsec
        __data = [time]

        for v in data.values:
            __data.append(v)

        if (time-self.angle_buffer[-1][0] < 1e-6):
            print(f"[logger] Spotted a repeated vel log")
            print(f"[logger] received data: {__data}")
            print(f"[logger] last data: {self.vel_buffer[-1]}")
            print(f"[logger] -> skipping the update")
            return

        # print("Logged vel:\n\t{}\n".format(__data))
        self.vel_buffer.append(__data)
        if len(self.vel_buffer)>self.buffer_size:
            del(self.vel_buffer[0])
    
    def __angle_callback(self, data):
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs
        
        time = float(__t_sec) + 1e-9 * __t_nsec
        __data = [time]

        for v in data.values:
            __data.append(v)

        if (time-self.angle_buffer[-1][0] < 1e-6):
            print(f"[logger] Spotted a repeated angle log")
            print(f"[logger] received data: {__data}")
            print(f"[logger] last data: {self.angle_buffer[-1]}")
            print(f"[logger] -> skipping the update")
            return

        # print("Logged angle:\n\t{}\n".format(__data))
        self.angle_buffer.append(__data)
        if len(self.angle_buffer)>self.buffer_size:
            del(self.angle_buffer[0])

    def __gyro_callback(self, data):
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs
        
        time = float(__t_sec) + 1e-9 * __t_nsec
        __data = [time]

        for v in data.values:
            __data.append(v)

        if (time-self.gyro_buffer[-1][0] < 1e-6):
            print(f"[logger] Spotted a repeated gyro log")
            print(f"[logger] received data: {__data}")
            print(f"[logger] last data: {self.gyro_buffer[-1]}")
            print(f"[logger] -> skipping the update")
            return

        # print("Logged gyro:\n\t{}\n".format(__data))
        self.gyro_buffer.append(__data)
        if len(self.gyro_buffer)>self.buffer_size:
            del(self.gyro_buffer[0])

    def log_run(self):
        # self.__read_launch_file()
        rospy.Subscriber('/cf4/pos', GenericLogData, self.__pos_callback)
        rospy.Subscriber('/cf4/vel', GenericLogData, self.__vel_callback)
        rospy.Subscriber('/cf4/angle', GenericLogData, self.__angle_callback)
        # rospy.Subscriber('/cf4/gyro', GenericLogData, self.__gyro_callback)
        rospy.spin()

    # def __del__(self):
        # self.csv_f.close()
        # print("destructor")

    def retrieve_state(self):
        # print("[Logger] Retrieving state from buffers:")
        # print("\t pos: {}".format(self.pos_buffer))
        # print("\t vel: {}".format(self.vel_buffer))
        # print("\t angle: {}".format(self.angle_buffer))
        state = np.zeros(12)
        # +1 since the first element in __data is timestamp
        for idx in range(3):
            state[idx] = self.pos_buffer[-1][idx+1]
        for idx in range(3):
            state[idx+3] = self.vel_buffer[-1][idx+1]
        # for idx in range(3):
        #     state[idx+3] = (self.pos_buffer[-1][idx+1] - self.pos_buffer[-2][idx+1])/(self.pos_buffer[-1][0] - self.pos_buffer[-2][0])
        for idx in range(3):
            state[idx+6] = self.angle_buffer[-1][idx+1]
        for idx in range(3):
            state[idx+9] = (self.angle_buffer[-1][idx+1] - self.angle_buffer[-2][idx+1])/(self.angle_buffer[-1][0] - self.angle_buffer[-2][0])
        # for idx in range(3):
        #     state[idx+9] = self.gyro_buffer[-1][idx+1]
        return state

    def retrieve_time(self, mode=None):
        if mode is None:
            return self.pos_buffer[-1][0]
        elif mode in ['oldest']:
            pos_time = self.pos_buffer[-1][0]
            vel_time = self.vel_buffer[-1][0]
            angle_time = self.angle_buffer[-1][0]
            # gyro_time = self.gyro_buffer[-1][0]
            # return min([pos_time, vel_time, angle_time, gyro_time])
            return min([pos_time, vel_time, angle_time])
            # return min([pos_time, angle_time])

    def retrieve_timestep(self, source='pos'):
        if source in 'pos':
            return self.pos_buffer[-1][0] - self.pos_buffer[-2][0]
        elif source in 'vel':
            return self.vel_buffer[-1][0] - self.vel_buffer[-2][0]
        elif source in 'angle':
            return self.angle_buffer[-1][0] - self.angle_buffer[-2][0]
        # elif source in 'gyro':
        #     return self.gyro_buffer[-1][0] - self.gyro_buffer[-2][0]
        else:
            print("specified source unknown")


class viconStateLogger():
    def __init__(self, vicon_buffer_size=50) -> None:
        self.vicon_buffer_size = vicon_buffer_size
        self.vicon_buffer = np.zeros((vicon_buffer_size, 13))
        self.vicon_idx = 0
        self.vicon2cf_translation = np.array([0., 0., 0.])

    def __vicon_callback(self, data):
        i_now = self.vicon_idx % self.vicon_buffer_size
        i_prev = (self.vicon_idx - 1) % self.vicon_buffer_size
        __t_sec = data.header.stamp.secs
        __t_nsec = data.header.stamp.nsecs

        time = float(__t_sec) + 1e-9 * __t_nsec

        x = data.transform.translation.x
        y = data.transform.translation.y
        z = data.transform.translation.z
        qx = data.transform.rotation.x
        qy = data.transform.rotation.y
        qz = data.transform.rotation.z
        qw = data.transform.rotation.w
        r,p,ya = Rotation.from_quat([qx, qy, qz, qw]).as_euler('XYZ', degrees=False)
        vx  = (x  - self.vicon_buffer[i_prev, 1]) / (time - self.vicon_buffer[i_prev, 0])
        vy  = (y  - self.vicon_buffer[i_prev, 2]) / (time - self.vicon_buffer[i_prev, 0])
        vz  = (z  - self.vicon_buffer[i_prev, 3]) / (time - self.vicon_buffer[i_prev, 0])
        rr  = (r  - self.vicon_buffer[i_prev, 7]) / (time - self.vicon_buffer[i_prev, 0])
        pr  = (p  - self.vicon_buffer[i_prev, 8]) / (time - self.vicon_buffer[i_prev, 0])
        yar = (ya - self.vicon_buffer[i_prev, 9]) / (time - self.vicon_buffer[i_prev, 0])
        self.vicon_buffer[i_now, :] = [time, x, y, z, vx, vy, vz, r, p, ya, rr, pr, yar]
        self.vicon_idx = self.vicon_idx + 1
        # print(f"{np.array_str(self.vicon_buffer[i_now, :], precision=1)}")


    def log_run(self):
        rospy.Subscriber('vicon/cf4/cf4', TransformStamped, self.__vicon_callback)
        rospy.spin()

    def retrieve_state(self):
        state = np.mean(self.vicon_buffer[:, 1:], axis=0)
        return state
    
    def retrieve_cf_state(self):
        state = self.retrieve_state()
        state[0:3] = state[0:3] + self.vicon2cf_translation
        return state

    def retrieve_time(self, mode=None):
        i_prev = (self.vicon_idx - 1) % self.vicon_buffer_size
        return self.vicon_buffer[i_prev, 0]

    def retrieve_timestep(self, source='pos'):
        i_prev = (self.vicon_idx - 1) % self.vicon_buffer_size
        i_prev_prev = (self.vicon_idx - 2) % self.vicon_buffer_size
        return self.vicon_buffer[i_prev, 0] - self.vicon_buffer[i_prev_prev, 0]


if __name__ == '__main__':
    # x = bufferStateLogger()
    x = viconStateLogger()
    rospy.init_node('Logger', anonymous=True)
    x.log_run()
