from PIL import Image, ImageDraw
import sys
import random

w, h = 50, 50

for x in range(500):

    blank = Image.new("RGB", (w,h), color = "black")

    p1 = (random.randint(5,45), random.randint(5,20))
    p2 = (random.randint(25,45), random.randint(20,45))
    p3 = (random.randint(5,25), random.randint(20,45))

    img = ImageDraw.Draw(blank)
    img.line((p1, p2), fill  = "white", width = 1) #Start (left, top) - End (left, top)
    img.line((p2, p3), fill  = "white", width = 1)
    img.line((p3, p1), fill  = "white", width = 1)

##blank.show()

    blank.save("Datasets/Triangles/%s.png"%x)