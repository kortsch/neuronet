from PIL import Image
import numpy as np
import numpy.random as rand

def f(x):
    



#%%
# Начальная инициализация весов случайными вещественными числами
w = np.random.sample((100, 100))
print(w[50,:])
for i in range(100):
    for j in range(100):
        w[i,j] = w[i,j]*2-1
print
print(w[50,:])
#%%





im = Image.open("r2.bmp")
#im.show()
width = im.size[0]
height = im.size[1]
print(width,height)
pix = im.load()
#for k in range(100):
#    print(pix[k,50])









