import sys, serial, glob
from time import sleep

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

ACQ = b'\x04'
RST = b'\x05'

class a2c_serial:
    def __init__(self):
        self.robot = serial.Serial()
    
    def autoSerial(self):
        if sys.platform.startswith('win'):
            ports = ['COM%s' % i for i in range(1,255)] # 1~257
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')
        for port in ports:
            try:
                self.robot = serial.Serial(port, baudrate=115200, timeout=0.01, write_timeout=0.01)
                self.robot.write(STX)
                reply = self.robot.read()
                if reply == ACK:
                    print('serial found')
                    self.robot.timeout=30
                    return 1
            except (OSError, serial.SerialException):
                print('An error occured')
        return 0
    

    def autoSerial_open(self):
        if not self.robot.isOpen():
            self.autoSerial()
            print(self.robot)
            if not self.robot.isOpen():
                print("Failed to open serial")
                return False
        print(self.robot.readline().decode())
        return True
    

    def reset(self):
        # initialize everythig for example angle
        # need sleep sometime to stabilization
        pass


    def step(self):
        # recieve angle, velocity, (current data?)
        num_byte = 0
        send_data = bytes([ACQ])
        self.robot.write(send_data)
        for _ in range(num_byte):
            recieve_data = [ord(x) for x in self.robot.readline()]
        