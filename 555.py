from PIL import Image, ImageDraw 
import numpy as np
image = Image.open('r4.bmp')  # Открываем изображение
draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования
width = image.size[0]  # Определяем ширину
height = image.size[1]  # Определяем высоту
pix = image.load()  # Выгружаем значения пикселей
a = np.empty(image.size[::-1], dtype = 'int')
print(a)
for x in range(width):
    for y in range(height):
       r = pix[x, y][0] #узнаём значение красного цвета пикселя
       g = pix[x, y][1] #зелёного
       b = pix[x, y][2] #синего
       sr = (r + g + b) // 3 #среднее значение
       a[y,x] = sr
       draw.point((x, y), (sr, sr, sr)) #рисуем пиксель

image.save("d:/Python/Lab5/result.jpg", "JPEG") #не забываем сохранить изображение
print(a)
image.show()



