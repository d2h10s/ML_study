import time
file_name = 'juingong'
i = 1
while True:
    print(file_name + str(i)+'.png')
    i = i + 1
    if i > 4:
        i = 1
    time.sleep(0.5)