# main.py
from cmu_graphics import *
from begin_page import drawBeginPage, beginMouseMove, beginMousePress, beginMouseRelease

def onAppStart(app):
    app.width = 1280
    app.height = 720
    app.bkg = 'Untitled_Artwork 3.png'
    app.title = 'Untitled_Artwork 4.png'
    app.page = 'home'  # home, begin, history

    app.started = False

    # Home button states
    app.beginHover = False
    app.beginPressed = False
    app.rectHeight = 75
    app.rectWidth = 300
    app.r = 37.5

def redrawAll(app):
    if app.page == 'home':
        drawHomePage(app)
    elif app.page == 'begin':
        drawBeginPage(app)

def drawHomePage(app):
    drawImage(app.bkg, 0, 0)
    drawImage(app.title, 0, 0)

    # Begin Button
    fillColor, yOffset = getButtonColorOffset(app.beginHover, app.beginPressed)
    drawRect(app.width/2 - app.rectWidth/2, app.height/2 + 25 + yOffset, app.rectWidth, app.rectHeight, fill=fillColor)
    drawCircle(app.width/2 - 150, app.height/2 + 63 + yOffset, app.r, fill=fillColor)
    drawCircle(app.width/2 + 150, app.height/2 + 63 + yOffset, app.r, fill=fillColor)
    drawLabel('BEGIN', app.width/2, app.height/2 + 60 + yOffset, size=40, bold=True)


def getButtonColorOffset(isHover, isPressed):
    if isPressed:
        return rgb(230, 150, 90), 4
    elif isHover:
        return rgb(255, 200, 140), -2
    else:
        return rgb(255, 182, 117), 0

def onMouseMove(app, mouseX, mouseY):
    if app.page == 'home':
        app.beginHover = pointInBeginButton(app, mouseX, mouseY)
    elif app.page == 'begin':
        beginMouseMove(app, mouseX, mouseY)

def onMousePress(app, mouseX, mouseY):
    if app.page == 'home':
        if pointInBeginButton(app, mouseX, mouseY):
            app.beginPressed = True
    elif app.page == 'begin':
        beginMousePress(app, mouseX, mouseY)

def onMouseRelease(app, mouseX, mouseY):
    if app.page == 'home':
        if app.beginPressed and pointInBeginButton(app, mouseX, mouseY):
            app.page = 'begin'
        app.beginPressed = False
        app.historyPressed = False
    elif app.page == 'begin':
        beginMouseRelease(app, mouseX, mouseY)

def pointInBeginButton(app, x, y):
    left = app.width/2 - app.rectWidth/2
    right = left + app.rectWidth
    top = app.height/2 + 25
    bottom = top + app.rectHeight
    inRect = left <= x <= right and top <= y <= bottom
    leftCx = app.width/2 - 150
    leftCy = app.height/2 + 63
    rightCx = app.width/2 + 150
    rightCy = app.height/2 + 63
    r = app.r
    inLeftCircle = ((x - leftCx)**2 + (y - leftCy)**2) <= r**2
    inRightCircle = ((x - rightCx)**2 + (y - rightCy)**2) <= r**2
    return inRect or inLeftCircle or inRightCircle


def main():
    runApp()

main()
