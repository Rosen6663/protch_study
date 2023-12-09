from torchvision.datasets import MNIST
from torchvision import transforms
mnist =  MNIST(root='data',train=True,download=True)


#print(mnist[0][0].show())
print(mnist[0])
ret = transforms.ToTensor()(mnist[0][0])
print(ret.size())