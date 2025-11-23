from cmu_graphics import *
import math

def onAppStart(app):
    app.width = 1280
    app.height = 720

    # State variable
    app.page = 'loading'  # 'loading', 'home', 'history', 'begin'

    # Loading animation
    app.loadingAngle = 0
    app.loadingRadius = 50
    app.loadingCenterX = app.width / 2
    app.loadingCenterY = app.height / 2
    app.loadingDots = 8  # number of dots in the spinner

    # Timing
    app.loadingFrames = 0  # count frames to switch page

def redrawAll(app):
    if app.page == 'loading':
        drawLoadingPage(app)
    elif app.page == 'home':
        drawHomePage(app)
    elif app.page == 'history':
        drawHistoryPage(app)

# ---------------- Loading Page ----------------
def drawLoadingPage(app):
    drawRect(0, 0, app.width, app.height, fill='black')
    drawLabel("LOADING...", app.width/2, app.height/2 - 100, size=50, bold=True, fill='white')

    # Spinner animation
    for i in range(app.loadingDots):
        angle = 2 * math.pi * i / app.loadingDots + app.loadingAngle
        x = app.loadingCenterX + app.loadingRadius * math.cos(angle)
        y = app.loadingCenterY + app.loadingRadius * math.sin(angle)
        alpha = int(255 * (i+1)/app.loadingDots)  # fade effect
        drawCircle(x, y, 15, fill=rgb(255, 255, 255, alpha))

def onStep(app):
    if app.page == 'loading':
        app.loadingAngle += 0.1  # rotate spinner
        app.loadingFrames += 1
        if app.loadingFrames > 120:  # after 120 frames (~2 seconds at 60fps)
            app.page = 'home'

# ---------------- Home Page ----------------
def drawHomePage(app):
    drawRect(0, 0, app.width, app.height, fill='lightblue')
    drawLabel("HOME PAGE", app.width/2, app.height/2, size=80, bold=True)

# ---------------- History Page ----------------
def drawHistoryPage(app):
    drawRect(0, 0, app.width, app.height, fill='lightgreen')
    drawLabel("HISTORY PAGE", app.width/2, app.height/2, size=80, bold=True)

def main():
    runApp()

main()
