import numpy as np
from NeuralNetwork import NeuralNetwork
from tkinter import *
from PIL import Image
from PIL import EpsImagePlugin
import os
from PIL import ImageOps

THETA = np.load("Theta28.npy", allow_pickle=True)
network = NeuralNetwork(THETA)
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.53.3\bin\gswin64c'

OLDX = None
OLDY = None
LINE_WIDTH = 50
COLOR = "black"


def calculate_guess():
    img = Image.open("digit.png")
    img_arr = np.asarray(img)
    img_arr = img_arr.flatten()
    img_arr = img_arr / 256
    img_arr = np.insert(img_arr, 0, 1, axis=0)
    img_arr = img_arr.reshape(1, 785)
    img_guess = network.calculate_output(THETA, img_arr)
    guess = np.argmax(img_guess)
    topl = Toplevel(root)
    lab = Label(topl, text="My guess is...")
    lab.pack()
    guess = Label(topl, text=guess)
    guess.pack()


def resize():
    img = Image.open("digit.png")
    img = img.resize((28, 28))
    inv_img = ImageOps.invert(img)
    inv_img = ImageOps.grayscale(inv_img)
    inv_img.save("digit.png")


def save_as_png(canvas, fileName):
    # save postscipt image
    canvas.postscript(file=fileName + '.eps')
    # use PIL to convert to PNG
    img = Image.open(fileName + '.eps')
    img.save(fileName + '.png', 'png')
    img.close()
    os.remove("digit.eps")
    resize()
    calculate_guess()


root = Tk()
root.title("Digit recognition")
c = Canvas(root, bg='white', width=600, height=600)
c.grid(row=1, columnspan=5)


def clear_canvas():
    c.delete("all")


save_button = Button(root, text="Evaluate", command=lambda: save_as_png(c, "digit"))
save_button.grid(row=0, column=1)

reset_button = Button(root, text="Reset cavas", command=clear_canvas)
reset_button.grid(row=0, column=3)


def paint(event):
    global OLDX, OLDY
    if OLDX and OLDY:
        c.create_line(OLDX, OLDY, event.x, event.y,
                      width=LINE_WIDTH, fill=COLOR,
                      capstyle=ROUND, smooth=TRUE, splinesteps=36)
    OLDX = event.x
    OLDY = event.y


def reset(event):
    global OLDX, OLDY
    OLDX, OLDY = None, None


c.bind('<B1-Motion>', paint)
c.bind('<ButtonRelease-1>', reset)

root.mainloop()
