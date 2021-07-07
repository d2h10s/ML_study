import sys, serial, glob, time
import numpy as np
from time import sleep

DEBUG_ON = False

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

ACQ = b'\x04'
RST = b'\x05'
GO_CW = b'\x70'
GO_CCW = b'\x71'

DEBUGGING = 1
COMMAND = 2

BAUDRATE = 115200

class a2c_serial:
    def __init__(self):
        self.ser = serial.Serial()
        self.port = None
        self.observation_space_n = 6
        self.action_space_n = 2
        self.temp_mx106 = 0
        self.temp_ahrs = 0
        self.wait_time = 140
        self.EPS = np.finfo(np.float32).eps.item()
        self.max_angle = 0
        self.th1 = 0
    
    def serial_open(self, target_port=None):
        if target_port==None:
            if sys.platform.startswith('win'):
                ports = ['COM%s' % i for i in range(1, 255)]  # 1~257
            elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
                # this excludes your current terminal "/dev/tty"
                ports = glob.glob('/dev/tty[A-Za-z]*')
            elif sys.platform.startswith('darwin'):
                ports = glob.glob('/dev/tty.*')
            else:
                raise EnvironmentError('Unsupported platform')
        else:
            ports = [target_port]
        print('searching serial device...')
        for port in ports:
            try:
                self.ser = serial.Serial(port, baudrate=BAUDRATE, timeout=1, write_timeout=1)
                reply, data_type = self.write_command(RST)
                if reply.startswith('STX,ACK') and data_type == COMMAND:
                    print(f'serial {port} found')
                    self.port = port
                    self.ser.timeout = 3
                    self.ser.write_timeout = 3
                    return True
                else:
                    print(f'serial {port} found but NAK', reply)
                    self.ser.close()
            except Exception as e:
                print(port, e)
        print('could not find serial device\n')
        return False

    def serial_close(self):
        self.ser.close()

    def write_command(self, command):
        if DEBUG_ON: print('start write')
        ret = self.ser.isOpen()
        while ret:
            self.ser.write(command)
            rx_buffer = bytearray()
            rx_byte = None
            byte_cnt = 0
            start_time = time.time()
            while rx_byte != ord('!') and time.time()-start_time < 10 and byte_cnt < 128:
                if self.ser.in_waiting:
                    rx_byte = ord(self.ser.read())
                    rx_buffer.append(rx_byte)
                    byte_cnt += 1
            if rx_byte == None:
                print(f'{hex(ord(command))} sent but received null byte')
            try:
                rx_string = rx_buffer.decode('utf-8')
            except :
                return "", -1
            if rx_string.startswith('@'):
                rx_string = rx_string[:-1]
                data_type = 1
            elif rx_string.startswith('STX'):
                rx_string = rx_string[:-1]
                data_type = 2
            else:
                data_type = -1
                print('could not recognize data:', rx_buffer)
            if DEBUG_ON: print('end write')
            return rx_string, data_type

    def reset(self):
        if DEBUG_ON: print('start reset')
        ret = self.ser.isOpen()
        while ret:
            reply, data_type = self.write_command(RST)
            if reply.startswith('STX,ACK') and data_type == COMMAND:
                print('wait for stabilization')
                start_time = time.time()
                elapsed_time = 0
                time_threshold = np.max([self.wait_time*140*4/np.pi*self.th1,10])
                while elapsed_time < time_threshold:
                    elapsed_time = time.time() - start_time
                    print(f'elapsed {elapsed_time:.2f}s and completed {elapsed_time/time_threshold*100:6.2f}%\r', end='')
                    sleep(1)
                if DEBUG_ON: print('end reset')
                self.max_angle = 0
                obs = self.get_observation()
                print(f'the temperature of ahrs:{self.temp_ahrs:5.1f}℃, mx106:{self.temp_mx106:5.1f}℃')
                return obs
            else:
                print('received unrecognized bytes', reply)

    def step(self, action):
        if DEBUG_ON: print('start step')
        ret = self.ser.isOpen()
        while ret:
            try:
                if action == 1:  # action 1 is go up (clock wise)
                    self.ser.write(GO_CW)
                elif action == 0:       # action 0 is go down (counter clock wise)
                    self.ser.write(GO_CCW)
                else:
                    print('action is out of range', action)
                if DEBUG_ON: print('end step')
                return self.get_observation()
            except Exception as e:
                self.ser.close()
                print("write error occurred in step function", e)
                return

    def get_observation(self):
        if DEBUG_ON: print('start obs')
        ret = self.ser.isOpen()
        while ret:
            rx_data, data_type = self.write_command(ACQ)
            if data_type == COMMAND and rx_data.startswith('STX,ACQ'):
                try:
                    # STX,ACQ,ROLL,VEL_ahrs,TEMP_ahrs,POS_mx106,VEL_mx106,TEMP_mx106
                    rx_data = rx_data.replace('STX,ACQ,', '').split(',')
                    roll = np.deg2rad(float(rx_data[0]))  # rad/s
                    ahrs_vel = float(rx_data[1]) / 0.393  # v -> w
                    ahrs_temp = float(rx_data[2])
                    mx106_pos = (2100 - float(rx_data[3])) * 0.088
                    mx106_vel = float(rx_data[4]) * np.pi / 30  # rpm -> rad/s
                    mx106_temp = float(rx_data[5])
                    #print('{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}'.format(roll, ahrs_vel, ahrs_temp, mx106_pos, mx106_vel, mx106_temp))
                    ret = False
                except Exception as e:
                    print(e, ' occurred with', rx_data, 'in obs function')
            else:
                print('could not recognize ', rx_data)
        self.th1 = roll
        sin_th1 = np.sin(roll)
        cos_th1 = np.cos(roll)
        sin_th2 = np.sin(mx106_pos)
        cos_th2 = np.cos(mx106_pos)
        vel_th1 = ahrs_vel
        vel_th2 = mx106_vel
        self.max_angle = np.abs(roll) if self.max_angle < np.abs(roll) else self.max_angle
        self.temp_ahrs = ahrs_temp
        self.temp_mx106 = mx106_temp
        observation = np.array([sin_th1, cos_th1, sin_th2, cos_th2, vel_th1, vel_th2], dtype=float)
        assert any(observation)
        if DEBUG_ON: print('end obs')
        return observation

    def get_temperature(self):
        return (self.temp_ahrs, self.temp_mx106)
        