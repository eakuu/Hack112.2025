from cmu_graphics import *
from begin_page import drawBeginPage
from loadingPage import drawLoadingPage

def onAppStart(app):
    #app.background = rgb(175, 227, 199)
    app.width = 1280
    app.height = 720
    app.bkg = 'https://cdn.discordapp.com/attachments/1413613759951405201/1441972824272011264/Untitled_Artwork_3.png?ex=6923bd57&is=69226bd7&hm=2ceb0a99fd199bf7725b87c1a1340a7320efe949c206fb65aeb3c515bf76bb23'
    app.title = 'https://cdn.discordapp.com/attachments/1413613759951405201/1441976739541680180/Untitled_Artwork_4.png?ex=6923c0fd&is=69226f7d&hm=67fda510154552e8276237bb796020a45d58f5477624693bb3dbbc2df5d4bdf9'
    app.rectHeight = 75
    app.rectWidth = 300
    app.r = 37.5

    app.page = 'home'
    
    app.beginHover = False
    app.beginPressed = False
    
    app.historyHover = False
    app.historyPressed = False
    
def redrawAll(app):
    if app.page == 'home':
        drawHomePage(app)
    elif app.page == 'begin':
        drawBeginPage(app)
    elif app.page == 'loading':
        drawLoadingPage(app)

def drawHomePage(app):
    drawImage(app.bkg, 0, 0)
    drawImage(app.title, 0, 0)
#   drawOval(100, 200, 600, 1000, fill = rgb(163, 196, 178))
#   title = drawLabel("Parkinson's Eye", app.width/2, app.height/2 - 100, size = 100, font = 'arial', italic = True, bold = True)

    # -------- begin button ---------
    if app.beginPressed:
        fillColor = rgb(230, 150, 90)
        yOffset = 4
    elif app.beginHover:
        fillColor = rgb(255, 200, 140)
        yOffset = -2
    else:
        fillColor = rgb(255, 182, 117)
        yOffset = 0
    
    drawRect(app.width/2 - app.rectWidth/2, app.height/2 + 25 + yOffset, 300, 75, fill = fillColor)
    drawCircle(app.width/2 - 150, app.height/2 + 63 + yOffset, 37.5, fill = fillColor)
    drawCircle(app.width/2 + 150, app.height/2 + 63 + yOffset, 37.5, fill = fillColor)
    drawLabel('BEGIN', app.width/2, app.height/2 + 60 + yOffset, size = 40, bold = True, fill = rgb(209, 132, 65))
    
    # -------- history button ---------

    if app.historyPressed:
        fillColor = rgb(230, 150, 90)
        yOffset = 4
    elif app.historyHover:
        fillColor = rgb(255, 200, 140)
        yOffset = -2
    else:
        fillColor = rgb(255, 182, 117)
        yOffset = 0
        
    drawRect(app.width/2 - app.rectWidth/2, app.height/2 + 120 + yOffset, 300, 75, fill = fillColor)
    drawCircle(app.width/2 - 150, app.height/2 + 158 + yOffset, 37.5, fill = fillColor)
    drawCircle(app.width/2 + 150, app.height/2 + 158 + yOffset, 37.5, fill = fillColor)
    drawLabel('HISTORY', app.width/2, app.height/2 + 158 + yOffset, size = 40, bold = True, fill = rgb(209, 132, 65))
    
def onMouseMove(app, mouseX, mouseY):
    app.beginHover = pointInBeginButton(app, mouseX, mouseY)
    app.historyHover = pointInHistoryButton(app, mouseX, mouseY)

def onMousePress(app, mouseX, mouseY):
    if pointInBeginButton(app, mouseX, mouseY):
        app.beginPressed = True
        
    if pointInHistoryButton(app, mouseX, mouseY):
        app.historyPressed = True
        
def onMouseRelease(app, mouseX, mouseY):
    if app.beginPressed and pointInBeginButton(app, mouseX, mouseY):
        print('begin clicked')
    if app.historyPressed and pointInHistoryButton(app, mouseX, mouseY):
        print('history clicked')
        
    app.beginPressed = False
    app.historyPressed = False
    
def pointInBeginButton(app, x, y):
    #----- rectangle bounds ----
    left = app.width/2 - app.rectWidth/2
    right = left + app.rectWidth
    top = app.height/2 + 25
    bottom = top + app.rectHeight
    
    inRect = (left <= x <= right and top <= y <= bottom)
    
    #----- left Circle ----
    leftCx = app.width/2 - 150
    leftCy = app.height/2 + 63
    rightCx = app.width/2 + 150
    rightCy = app.height/2 + 63
    r = app.r
    
    inLeftCircle = ((x - leftCx)**2 + (y - leftCy)**2) <= r**2
    inRightCircle = ((x - rightCx)**2 + (y - rightCy)**2) <= r**2
    
    return inRect or inLeftCircle or inRightCircle
    
def pointInHistoryButton(app, x, y):
    #----- rectangle bounds ----
    left = app.width/2 - app.rectWidth/2
    right = left + app.rectWidth
    top = app.height/2 + 120
    bottom = top + app.rectHeight
    
    inRect = (left <= x <= right and top <= y <= bottom)
    
    #----- left Circle ----
    leftCx = app.width/2 - 150
    leftCy = app.height/2 + 158
    rightCx = app.width/2 + 150
    rightCy = app.height/2 + 158
    r = app.r
    
    inLeftCircle = ((x - leftCx)**2 + (y - leftCy)**2) <= r**2
    inRightCircle = ((x - rightCx)**2 + (y - rightCy)**2) <= r**2
    
    return inRect or inLeftCircle or inRightCircle

def main():
    runApp()

main()
