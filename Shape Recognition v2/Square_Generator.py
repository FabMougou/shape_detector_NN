from PIL import Image, ImageDraw
import sys
import random

w, h = 50, 50

for x in range(500):

    blank = Image.new("RGB", (w,h), color = "black")

    p1 = (random.randint(5,20), random.randint(5,20))
    p2 = (random.randint(30,45), random.randint(5,20))
    p3 = (random.randint(5,20), random.randint(30,45))
    p4 = (random.randint(30,45), random.randint(30,45))

    img = ImageDraw.Draw(blank)
    img.line((p1,p2), fill = "white", width = 1)
    img.line((p2,p4), fill = "white", width = 1)
    img.line((p4,p3), fill = "white", width = 1)
    img.line((p3,p1), fill = "white", width = 1)

    #blank.show()

    blank.save("Datasets/Squares/%s.png"%x)