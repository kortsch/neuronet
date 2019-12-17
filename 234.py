import random
from PIL import Image
import numpy as np
#mode = int(input('mode:')) #Считываем номер преобразования. 
image = Image.open("c:/download/test/08.bmp") #Открываем изображение.
width = image.size[0] #Определяем ширину. 
height = image.size[1] #Определяем высоту. 
print(width, height)	
pix = image.load() #Выгружаем значения пикселей.
#Npix = np.array(pix)
#print(Npix[50,50])
p0 = 0
p1 = 0
for j in range(height):
    if (j%100==0):
        for i in range(width):
            if (i%100==0):
                print(pix[i,j], end = ' ')
        print()    
                
        
        
#%%
import random
from PIL import Image
import numpy as np
import math
Ftarget = open("d:/!!!/target.txt")
print(Ftarget)
t = Ftarget.readlines()
print(t)
for i in range(len(t)):
    t[i]=t[i][:-1]
    t[i] = int(t[i])
print(t)
Xsize = 100
Ysize = 100
w = [[2*random.random()-1 for i in range(Xsize)] for j in range(Ysize)]
wnp = np.array(w)
#print(type(w[20,20]))
#print(wnp[1,1])




def perseptron(p, w):
    #s = 0
    #for i in range(Xsize):
    #    for j in range(Ysize):
    #        s += w[i][j]*pix[i,j]
    return w.sum()





Nteach = 1
for m in range(Nteach):
    k = random.randint(10,13)
    #print(k)
    pt = "d:/!!!/f"+str(k)+".bmp"
    print(k, pt, t[k-10])
    image = Image.open(pt)
    #width = image.size[0] #Определяем ширину. 
    #height = image.size[1] #Определяем высоту.       
    #print(width, height)	
    pix = image.load() #Выгружаем значения пикселей.  
    #pix = np.array(pix)
    print(type(pix)) 
    print(pix[20,20])  
    image.close()
    s = 0
    #for i in range(Xsize):
    #    for j in range(Ysize):
    #        s += w[i][j]*pix[i,j]
    #y = perseptron(pix, wnp)
    #print(y)
    #educat(pix, w, y, t)

    
#%%        
        
        


#%%
        
        
        
#%%
        
        
        

        
