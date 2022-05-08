#!/usr/bin/env python

import time
import pygame
from pygame.locals import *
import sys, os
if sys.platform == 'win32' or sys.platform == 'win64':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()

Screen = max(pygame.display.list_modes())
icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
pygame.display.set_caption("[Program] - [Author] - [Version] - [Date]")
Surface = pygame.display.set_mode(Screen,FULLSCREEN)

black = 0,0,0
red = 255,0,0
white = 255,255,255
green = 0,75,0
orange = 175,75,0

Font = pygame.font.Font("font.ttf",1000)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

test = Font.render("0",True,(255,255,255))
width = test.get_width()
height = test.get_height()
totalwidth = 4.5 * width
timerCountDown = 1
timerCountUp = 1
preTimerCountdown = 1

def quit():
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    pygame.quit(); sys.exit()
def GetInput():
    key = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE: quit()


def CountUp(startTime,timeDuration,backgroundColor):

    Surface.fill(backgroundColor)
    start_pos = (Screen[0]/2)-(totalwidth/2)
    currentTime = time.time()
    elapsedTime = currentTime - startTime

    displayTime = time.strftime('%M:%S', time.gmtime(elapsedTime)) #'%H:%M:%S'
    pos = [start_pos,(Screen[1]/2)-(height/2)]
    timeDuration = time.strftime('%M:%S', time.gmtime(timeDuration))
    Surface.blit(Font.render(displayTime,True,(white)),pos)
    pygame.display.flip()

    if displayTime == timeDuration:
        time.sleep(0)
        #quit()
        global timerCountUp
        timerCountUp = 0
    startTime = currentTime

def CountDown(startTime,timeDuration,backgroundColor):

    Surface.fill(backgroundColor)
    startTime = startTime +1
    start_pos = (Screen[0]/2)-(totalwidth/2)

    currentTime = time.time()
    elapsedTime = currentTime - startTime
    displayTime = timeDuration - elapsedTime
    displayTime = time.strftime('%M:%S', time.gmtime(displayTime)) #'%H:%M:%S'
    pos = [start_pos,(Screen[1]/2)-(height/2)]
    timeDuration = time.strftime('%M:%S', time.gmtime(timeDuration + 1))

    Surface.blit(Font.render(displayTime,True,(white)),pos)
    pygame.display.flip()

    if displayTime == "00:00":
        global timerCountDown
        timerCountDown = 0
    startTime = currentTime

def main():
    startTime = time.time()
    Clock = pygame.time.Clock()
    global timerCountUp
    timerCountUp = 1
    global timerCountDown
    timerCountDown = 1
    global preTimerCountdown
    preTimerCountdown = 1

    while True:
        GetInput()
        Clock.tick(60)
        while timerCountUp != 0:
            CountUp(startTime,7,green)
            GetInput()
            global timerCountDown
            timerCountDown = 1
        startTime = time.time()
        while timerCountDown != 0:
            CountDown(startTime,3,orange)
            GetInput()
            global timerCountUp
            timerCountUp = 1
        startTime = time.time()

if __name__ == '__main__': main()