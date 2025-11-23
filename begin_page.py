# begin_page.py
from cmu_graphics import *

def drawBeginPage(app):
    drawImage('Untitled_Artwork 3.png', 0, 0)
    drawLabel('To start, press begin:', app.width/2, app.height/2 - 50, size=60, bold=True)

    if not app.started:
        # Draw the begin button
        if app.beginPressed:
            fillColor = rgb(230, 150, 90)
            yOffset = 4
        elif app.beginHover:
            fillColor = rgb(255, 200, 140)
            yOffset = -2
        else:
            fillColor = rgb(255, 182, 117)
            yOffset = 0

        drawRect(app.width/2 - app.rectWidth/2, app.height/2 + 25 + yOffset, 300, 75, fill=fillColor)
        drawCircle(app.width/2 - 150, app.height/2 + 63 + yOffset, 37.5, fill=fillColor)
        drawCircle(app.width/2 + 150, app.height/2 + 63 + yOffset, 37.5, fill=fillColor)
        drawLabel('BEGIN', app.width/2, app.height/2 + 60 + yOffset, size=40, bold=True, fill=rgb(209, 132, 65))
    else:
        # After pressed
        drawLabel('Press the button on the device', app.width/2, app.height/2 + 30, size=30, bold=True)
        drawLabel('to start recording:', app.width/2, app.height/2 + 70, size=30, bold=True)

def beginMouseMove(app, mouseX, mouseY):
    app.beginHover = pointInBeginButton(app, mouseX, mouseY)

def beginMousePress(app, mouseX, mouseY):
    if pointInBeginButton(app, mouseX, mouseY):
        app.beginPressed = True

def beginMouseRelease(app, mouseX, mouseY):
    if app.beginPressed and pointInBeginButton(app, mouseX, mouseY):
        app.started = True
        print("Begin button clicked in Begin page")
    app.beginPressed = False

def pointInBeginButton(app, x, y):
    left = app.width/2 - app.rectWidth/2
    right = left + app.rectWidth
    top = app.height/2 + 25
    bottom = top + app.rectHeight
    inRect = (left <= x <= right and top <= y <= bottom)
    leftCx = app.width/2 - 150
    leftCy = app.height/2 + 63
    rightCx = app.width/2 + 150
    rightCy = app.height/2 + 63
    r = app.r
    inLeftCircle = ((x - leftCx)**2 + (y - leftCy)**2) <= r**2
    inRightCircle = ((x - rightCx)**2 + (y - rightCy)**2) <= r**2
    return inRect or inLeftCircle or inRightCircle
