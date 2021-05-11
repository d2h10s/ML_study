'''
def isPrime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

n = int(input())


if isPrime(n):
    print('Prime')
else:
    print('not Prime')
'''
a = 5
print(a.bit_length())