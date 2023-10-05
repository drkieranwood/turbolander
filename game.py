import numpy as np
import pyglet
import pygame


from globals import *
from shapeobjects import *
from additionalMathMethods import *

#Helper for 2D vectors
vec2 = pygame.math.Vector2



class Game:
    def __init__(self):
        bgImg = pyglet.image.load('images/city.png')
        self.trackSprite = pyglet.sprite.Sprite(bgImg, x=0, y=0)

        #Init. drone
        self.drone = Drone(self.walls)


class Drone:
    def __init__(self, walls):
        global vec2
        #starting state
        self.x = startX    
        self.y = startY
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0

        #drone collision size
        self.width = 40
        self.height = 20

        #drone physics/agility
        self.turningRate = 5.0 / self.width
        self.dragcoefficient = 0.98
        # self.maxSpeed = self.width / 4.0
        #TODO....add more from 2D equations of motion

        #sensing
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.dronePic = pyglet.image.load('images/drone.png')
        self.droneSprite = pyglet.sprite.Sprite(self.dronePic, x=self.x, y=self.y)    #render at the start location, size, and orientation
        self.droneSprite.update(rotation=0, scale_x=self.width / self.droneSprite.width,
                              scale_y=self.height / self.droneSprite.height)
        
        #state flags
        self.dead = False
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        #init the walls and rewards
        self.walls = walls
        #self.rewardGates = rewards

        self.reward = 0
        self.score = 0
        self.lifespan = 0


    def reset(self):
        global vec2
        #starting state
        self.x = startX    
        self.y = startY
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        #TODO...add more state resets here

        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.dead = False
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        self.reward = 0
        self.lifespan = 0           #a counter for the 'duration' a drone existed before a failure is detected
        self.score = 0  


    def show(self):
        # first calculate the center of the drone in order to allow the
        # rotation of the drone to be anchored around the center
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.droneSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.droneSprite.draw()
        # self.showCollisionVectors()


    #For some debug purposes to draw an overlay of critical information
    # def showCollisionVectors(self):
    #     global drawer
    #     for point in self.lineCollisionPoints:
    #         drawer.setColor([255, 0, 0])
    #         drawer.circle(point.x, point.y, 5)


    def getPositionOnCarRelativeToCenter(self, right, up):
        global vec2
        w = self.width
        h = self.height
        rightVector = vec2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()
        return vec2(self.x, self.y) + ((rightVector * right) + (upVector * up))

    
    def updateWithAction(self, actionNo):
        #Convert an action number into its physical meaning for the drone motion.
        #This is probably the AI interface to the drone control (just eight different commands)
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if actionNo == 0:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 2:
            self.accelerating = True
        elif actionNo == 3:
            self.reversing = True
        elif actionNo == 4:
            self.accelerating = True
            self.turningLeft = True
        elif actionNo == 5:
            self.accelerating = True
            self.turningRight = True
        elif actionNo == 6:
            self.reversing = True
            self.turningLeft = True
        elif actionNo == 7:
            self.reversing = True
            self.turningRight = True
        elif actionNo == 8:
            pass

        totalReward = 0

        #If the drone has not collided, update its position based on the model and current input
        for i in range(1):
            if not self.dead:
                self.lifespan+=1
                self.updateControls()       #adds some more natural command responses
                self.move()

                #After motion update, check if a failure/collision occured
                if self.hitAnObject():
                    self.dead = True
                # self.checkRewardGates()
                totalReward += self.reward

        #If drone is alive, setup its new sensed inputs vectors
        self.setVisionVectors()
        self.reward = totalReward


    #Called every frame
    def update(self):
        if not self.dead:
            self.updateControls()           #adds some more natural command responses
            self.move()

            if self.hitAnObject():
                self.dead = True
            # self.checkRewardGates()
            self.setVisionVectors()


    #Update position based on model physics and previous state
    def move(self):
        global vec2
        #TODO...all of this section needs to be based on equations of motion for flight
        self.vel += self.acc
        self.vel *= self.dragcoefficient

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.y += self.vel * self.direction.y

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        #Move the position state along one tick
        #TODO...this will all be flight equations of motion
        self.x += addVector.x
        self.y += addVector.y


    #Change the drone state based on user inputs
    #Seems to be some sort of system to make the movement more natural to user inputs, i.e. turning rate 
    #    changes based on speed, acceleration to change direction is more agressive than just getting faster
    #    in the same direction (i.e. brakes)
    #TODO...a lot of this will be superceeded by the implementation of the equations of motion and might be removed
    def updateControls(self):
        multiplier = 1
        if abs(self.vel) < 5:
            multiplier = abs(self.vel) / 5
        if self.vel < 0:
            multiplier *= -1

        driftAmount = self.vel * self.turningRate * self.width / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if self.turningLeft:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)

        elif self.turningRight:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)

        self.acc = 0
        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed
        elif self.reversing:
            if self.vel > 0:
                self.acc = -3 * self.accelerationSpeed
            else:
                self.acc = -1 * self.accelerationSpeed


    def hitAnObject(self):
        #Iterate all wall objects and check if drone has collided
        for wall in self.walls:
            if wall.hitDrone(self):
                return True

        return False


#Immovable objects the drone must not hit 
class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(2)

    """
    draw the line
    """
    def draw(self):
        self.line.draw()

    """
    returns true if the drone object has hit this wall
    """
    def hitDrone(self, car):
        global vec2
        cw = drone.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False

#eof