import serial, sys, glob

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'
robot = serial.Serial()

def autoSerial():
    global robot
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
            robot = serial.Serial(port, baudrate=115200, timeout=0.01, write_timeout=0.01)
            robot.write(STX)
            reply = robot.read()
            if reply == ACK:
                print('serial found')
                robot.timeout=30
                return 1
        except (OSError, serial.SerialException):
            print('An error occured')
    return 0

while True:
    if not robot.isOpen():
        autoSerial()
        print(robot)
        if not robot.isOpen():
            print("Failed to open serial")
            break
    print(robot.readline().decode())