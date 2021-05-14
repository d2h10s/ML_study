import threading
from threading import Lock
import time

def cal1():
    global a
    for i in range(1000000):
        l.acquire()
        a += 3
        l.release()
        print('영상 처리 완료', a)

t = threading.Thread(target=cal1)
l = Lock()
t.start()
a = 3
while True:
    for i in range(100):

        print('API 받아옴', a)