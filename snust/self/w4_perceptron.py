import numpy as np
import matplotlib.pyplot as plt

class perceptron:
    def __init__(self, clo, clx):
        self.clo = clo
        self.clx = clx
        self.data_dim = len(self.clo[0])
        self.bias = np.random.normal()
        self.weights = np.random.normal(size=self.data_dim)
        
    def forward(self, pt):
        return self.bias + np.dot(self.weights, pt) 
    
    def predict(self, pt):
        if self.forward(pt) > 0:
            return 1
        else:
            return -1
    
    def visualize(self, db_op = False):
        assert self.data_dim ==2
        
        plt.figure(figsize=(5,5))
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
    def __init__(self, perceptron):
        self.pcr = perceptron
        self.clo = self.pcr.clo
        self.clx = self.pcr.clx
        print("테스트 데이터 분류 성공 퍼센트는 %.2f%%입니다." %self.test()[1])
        
        
    def test(self):
        s_count = 0
        self.incorrect = []
        for pt in self.clo + self.clx:
            t = 2*int(pt in self.clo) - 1
            y = self.pcr.predict(pt)
            if t*y > 0:
                s_count +=1
            else:
                self.incorrect.append(pt)
        self.ratio = 100*s_count / (len(self.clo+self.clx))
        return self.incorrect, self.ratio
    
    
    def update(self):
        self.test()
        if self.incorrect == []:
            print("모든 테스트 데이터 분류 성공.")
            return False
        else:
            miss_pt = np.random.choice(self.incorrect)
            t = 2*int(miss_pt in self.clo) -1
            self.pcr.weights += t * np.array(miss_pt)
            self.pcr.bias += t
            return True
        
    def iterative(self):
        step = 1
        while self.update() and step<50:
            print("%d번째 update 결과 분류 성공율은 %.2f" %(step, self.test()[1]))
            step +=1
        pass
        

aa = [[-1,0],[0,1]]
bb = [[1,0], [1,1]]
p1 = perceptron(aa,bb)

p1.predict([1,1])

p1.visualize(True)

ff = trainer(p1)
ff.iterative()

p1.visualize(True)