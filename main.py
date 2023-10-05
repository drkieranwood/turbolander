#kieran.wood@manchester.ac.uk
#01/09/2023
#TurboLander v1.
#An exploration of AI-based control of multi-copter landing trajectories.
#Quite a lot inspired by, and copied from, CodeBullet "A.I. Learns to DRIVE" https://github.com/Code-Bullet/Car-QLearning

#When modifying, download and run pipreqs to generate the submodule requirements
#pip install pipreqs
#pipreqs .
#(watch out that numpy is compatible with tensorflow. Might need manual intervention)

#To clone and run, first install dependencies
#pip install -r requirements.txt


#Python libs
import pyglet                   #The game rendering engine
from pyglet.window import key   #For key presses interaction
import pygame
import tensorflow as tf         #Deep learning library
import numpy as np              #Handle matrices
from collections import deque
import random
import os
import math

#The program settings
from globals import *           #Program settings
from game import Game           #The main code controlling the game physics

class QLearning:
    def __init__(self, game):
        pass

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        #Set background color
        backgroundColor = [0, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        glClearColor(*backgroundColor)

        #Load the game
        self.game = Game()
        self.ai = QLearning(self.game)


    def on_close(self):
        self.ai.sess.close()

    #When key is hit
    def on_key_press(self, symbol, modifiers):
        # pass
        if symbol == key.RIGHT:
            self.game.drone.turningRight = True
        
        if symbol == key.LEFT:
            self.game.drone.turningLeft = True
        
        if symbol == key.UP:
            self.game.drone.accelerating = True
        
        if symbol == key.DOWN:
            self.game.drone.reversing = True

    #When key is released
    def on_key_release(self, symbol, modifiers):
        # pass
        if symbol == key.RIGHT:
            self.game.drone.turningRight = False
        
        if symbol == key.LEFT:
            self.game.drone.turningLeft = False
        
        if symbol == key.UP:
            self.game.drone.accelerating = False
        
        if symbol == key.DOWN:
            self.game.drone.reversing = False
        
        if symbol == key.SPACE:
            self.ai.training = not self.ai.training
    
    #Mouse click
    def on_mouse_press(self, x, y, button, modifiers):
        pass
        # # print(x,y)
        # if self.firstClick:
        #     self.clickPos = [x, y]
        # else:
        #     # print("self.walls.append(Wall({}, {}, {}, {}))".format(self.clickPos[0],
        #     #                                                        displayHeight - self.clickPos[1],
        #     #                                                        x, displayHeight - y))
        #
        #     # self.gates.append(RewardGate(self.clickPos[0], self.clickPos[1], x, y))
        #
        # self.firstClick = not self.firstClick

    #On window resize
    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    #Every frame
    def on_draw(self):
        self.game.render()

    #Every frame
    def update(self, dt):
        #train x5 times in a row?
        for i in range(5):
            if self.ai.training:
                self.ai.train()
            else:
                self.ai.test()
                return


if __name__ == "__main__":
    window = MyWindow(displayWidth, displayHeight, "TurboLander", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()
    
    
#eof