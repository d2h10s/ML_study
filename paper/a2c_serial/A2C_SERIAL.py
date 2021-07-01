import sys, serial, glob
import numpy as np
from time import sleep

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

ACQ = b'\x04'
RST = b'\x05'
GO_CW = b'\x70'
GO_CCW = b'\x71'


class a2c_serial:
    def __init__(self):
        self.ser = serial.Serial()
        self.port = None
        self.observation_space_n = 6
        self.action_space_n = 2
        self.temp_mx106 = 0
        self.temp_ahrs = 0
    
    def serial_open(self):
        if sys.platform.startswith('win'):
            ports = ['COM%s' % i for i in range(1, 255)]  # 1~257
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')
        print('searching serial device...')
        for port in ports:
            try:
                self.ser = serial.Serial(port, baudrate=115200, timeout=1, write_timeout=1)
                self.ser.write(RST)
                reply = self.ser.readline().decode('utf-8')
                if reply.startswith('STX,ACK'):
                    print(f'serial {port} found')
                    self.port = port
                    self.ser.timeout = 30
                    return True
                else:
                    print(f'serial {port} found but NAK', reply)
            except (serial.SerialException):
                pass
        print('could not find serial device')
        return False

    def reset(self):
        self.ser.write(RST)
        reply = self.ser.readline().decode('utf-8')
        if reply.startswith('STX,ACK'):
            return True
        else:
            return False
        sleep(180)

    def step(self, action):
        if action:  # action 1 is go up (clock wise)
            self.ser.write(GO_CW)
        else:       # action 0 is go down (counter clock wise)
            self.ser.write(GO_CCW)
        return self.get_observation()

    def get_observation(self):
        while ret:
            # ser.write(GO_MIN)
            self.ser.write(ACQ)
            rx_data = self.ser.readline().decode('utf-8')
            if rx_data.startswith('@'):
                print(rx_data, end='')
            elif rx_data.startswith('STX,ACQ'):
                try:
                    # STX,ACQ,ROLL,VEL_ahrs,TEMP_ahrs,POS_mx106,VEL_mx106,TEMP_mx106
                    rx_data = rx_data.replace('STX,ACQ,', '').split(',')
                    roll = np.deg2rad(float(rx_data[0]))  # rad/s
                    ahrs_vel = float(rx_data[1]) / 0.393  # v -> w
                    ahrs_temp = float(rx_data[2])
                    mx106_pos = (2100 - float(rx_data[3])) * 0.088
                    mx106_vel = float(rx_data[4]) * np.pi / 30  # rpm -> rad/s
                    mx106_temp = float(rx_data[5])
                    print('{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}'.format(roll, ahrs_vel, ahrs_temp, mx106_pos,
                                                                                   mx106_vel, mx106_temp))
                except Exception as e:
                    print(e, ' occurred with', rx_data)
                ret = False
            else:
                print('could not recognize ', rx_data)
        sin_th1 = np.sin(roll)
        cos_th1 = np.cos(roll)
        sin_th2 = np.sin(mx106_pos)
        cos_th2 = np.cos(mx106_pos)
        vel_th1 = ahrs_vel
        vel_th2 = mx106_vel
        self.temp_ahrs = ahrs_temp
        self.temp_mx106 = mx106_temp
        return [sin_th1, cos_th1, sin_th2, cos_th2, vel_th1, vel_th2]

    def get_temperature(self):
        return (self.temp_ahrs, self.temp_mx106)
        