import sys, glob, serial
import numpy as np
from time import sleep
STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'
RDY = b'\x16'
MER = b'\x17'
SER = b'\x18'

ACQ = b'\x04'
RST = b'\x05'
GO_MIN = b'\x70'
GO_MAX = b'\x71'

def autoSerial():
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
    ser = serial.Serial()
    for port in ports:
        try:
            ser = serial.Serial(port, baudrate=115200, timeout=1, write_timeout=1)
            ser.write(RST)
            reply = ser.readline().decode('utf-8')
            if reply.startswith('STX,ACK'):
                print(f'serial {port} found')
                ser.timeout = 30
                return (ser, True)
            else:
                print(f'serial {port} found but NAK', reply)
        except (serial.SerialException):
            pass
    print('could not find serial device')
    return (ser, False)

ser, ret = autoSerial()
while ret:
    #ser.write(GO_MIN)
    ser.write(ACQ)
    rx_data = ser.readline().decode('utf-8')
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