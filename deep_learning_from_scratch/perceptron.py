import numpy as np
import random
import matplotlib.pyplot as plt

class perceptron:
    def __init__(self, clo, clx):
        self.clo = clo
        self.clx = clx
        self.data_dim = len(self.clo[0])
        self.bias = np.random.normal()
        self.weights = np.random.normal(size=self.data_dim)

    def __forward(self, pt):
        return self.bias + self.weights @ pt
    
    def predict(self, pt):
        if self.__forward(pt) > 0:
            return 1
        else:
            return -1
    
    def visualize(self, db_op = False):
        assert self.data_dim ==2
        
        plt.figure(figsize=(5,5))
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        for pt in self.clo:
            plt.plot(pt[0], pt[1], 'ro', markersize = 20)
        for pt in self.clx:
            plt.plot(pt[0],pt[1], 'bx', markersize = 20)
    
        if db_op:
            xmin = min([pt[0] for pt in self.clx + self.clo])
            xmax = max([pt[0] for pt in self.clx + self.clo]) 
    
            xrange = np.linspace(xmin,xmax, int((xmax-xmin)*100))
            w1 = self.weights[0]
            w2 = self.weights[1]
            b  = self.bias
            line = lambda x : - w1*x / w2 - b / w2 
            y = line(xrange)
            max_y = max(np.max(y) , max([pt[1] for pt in self.clo+self.clx]) )
            plt.plot(xrange, y , 'k--', lw=5)
            plt.fill_between(xrange, y, [max_y for _ in xrange], alpha=0.5, color='blue')
        return plt.show()
    
class trainer:
    def __init__(self, perceptron:perceptron):
        self.pcr = perceptron
        self.clo = self.pcr.clo
        self.clx = self.pcr.clx
        print("테스트 데이터의 분류 성공 퍼센트는 {}%%입니다.".format(self.test()[1]))
    
    def test(self):
        s_count = 0
        self.incorrect = []
        for pt in self.clo + self.clx:
            t = 2*(pt in self.clo) - 1
            y = self.pcr.predict(pt)
            if t*y > 0:
                s_count += 1
            else:
                self.incorrect.append(pt)
        self.ratio = 100 * s_count / len(self.clo+self.clx)
        return self.incorrect, self.ratio

    def update(self):
        self.test()
        print('b[{}] w1[{}] w2[{}]'.format(self.pcr.bias, self.pcr.weights[0], self.pcr.weights[1]))
        print('slope[{}] bias[{}]'.format(-self.pcr.weights[0]/self.pcr.weights[1], -self.pcr.bias/self.pcr.weights[1]))
        self.pcr.visualize(True)
        if self.incorrect == []:
            print('모든 데이터 분류 성공')
            return False
        else:
            miss_pt = random.choice(self.incorrect)
            t = 2*(miss_pt in self.clo) - 1
            self.pcr.weights += t * np.array(miss_pt)
            self.pcr.bias += t
            return True
    
    def iterative(self):
        step = 1
        while self.update() and step<50:
            print("{}번째 update 결과 분류 성공율은 {}".format(step, self.test()[1]))
            step += 1
    

    

clx = [[0,1],[1,1],[1,0]]
clo = [[0,0],[-1,-1]]
pcr = perceptron(clo, clx)
train = trainer(pcr)
train.iterative()
pcr.visualize(True)