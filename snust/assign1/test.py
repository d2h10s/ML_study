import torch
from torch import nn
class net(torch.nn.Module):

    def __init__(self):
        super(net, self).__init__()
        
        #input (num_batch, channel=1, 28, 28)
        #생각한 신경망 모델로 변환할 것. (Conv2d, Linear 조합 이용.)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 4, stride=1, padding=0)
        self.conv3.
        self.fc    = nn.Linear((128, 3) 
        
    
    
    def forward(self, x): #구현한 신경망 모델에 따라 적절히 변형할 것
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
    
    def predict(self, image): #test image는 넘파이 (28,28) 형상 2d 배열임. 테스트 이미지 1장을 입력했을 때 결과가 출력되도록 할 것.
        image = (image -[128])/128
        image = torch.from_numpy(image).float()
        image = image.view(-1,1, 28,28)
        out   = self.forward(image).view(-1)
        return out.argmax().item()
    
    def accuracy(self, input_data, target_data):
        predict = self.forward(input_data).argmax(axis=1)
        correct_cnt = sum(predict==target_data)
        return correct_cnt.item()/len(target_data)
    
    def save(self, name):
        torch.save(self.state_dict(), name + '.pt')
        print('Models saved successfully')
    
    def load(self, name):
        self.load_state_dict(torch.load(name + '.pt'))
        print ('Models loaded succesfully')
    


