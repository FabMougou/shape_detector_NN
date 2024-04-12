from PIL import Image, ImageDraw
import sys
import random


w, h = 50, 50
for x in range(500):
    r1 = random.randint(5,15) #How far from left
    r2  = random.randint(5,15) #How far from top
    r3 = random.randint(5,15) #How far from right
    r4 = random.randint(5,15) #How far from bottom

    shape = [(r1, r2), (w-r3, h-r4)]

    blank = Image.new("RGB", (w, h), color = "black")

    img = ImageDraw.Draw(blank)
    img.arc(shape, start = 0, end = 360, fill = "white")
    #blank.show()

    blank.save("Datasets/Circles/%s.png" %x)


