from torchvision.datasets import MNIST

mnist =  MNIST(root='data',train=True,download=True)


#print(mnist[0][0].show())
print(mnist[0])