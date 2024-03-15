from graphics import *
from gobang_class import *


GRID_WIDTH = 40


def gobangwin():
    win = GraphWin("this is a gobang game", GRID_WIDTH * COLUMN, GRID_WIDTH * ROW)#界面框
    win.setBackground("yellow")#背景颜色
    i1 = 0

    while i1 <= GRID_WIDTH * COLUMN:#画竖线
        l = Line(Point(i1, 0), Point(i1, GRID_WIDTH * COLUMN))
        l.draw(win)
        i1 = i1 + GRID_WIDTH
    i2 = 0

    while i2 <= GRID_WIDTH * ROW:#画横线
        l = Line(Point(0, i2), Point(GRID_WIDTH * ROW, i2))
        l.draw(win)
        i2 = i2 + GRID_WIDTH
    return win


def run():
    gb = GoBang(AI_first=False)
    gb.initialize()

    win = gobangwin()


    change = 0

    while gb.g == 0:

        if change % 2 == 1:#第1/3/5...步，轮到AI下
            pos = gb.AIRun()

            if pos in list3:#如果无路可走，棋局结束
                message = Text(Point(200, 200), "不可用的位置" + str(pos[0]) + "," + str(pos[1]))
                message.draw(win)
                gb.g = 1

            piece = Circle(Point(GRID_WIDTH * pos[0], GRID_WIDTH * pos[1]), 16)
            piece.setFill('white')
            piece.draw(win)

            if game_win(list1):
                message = Text(Point(100, 100), "white win.")
                message.draw(win)
                gb.g = 1

            change = change + 1


        else:#第0/2/4步，轮到人下棋，使用getMouse
            p2 = win.getMouse()

            if not ((round((p2.getX()) / GRID_WIDTH), round((p2.getY()) / GRID_WIDTH)) in list3):

                a2 = round((p2.getX()) / GRID_WIDTH)
                b2 = round((p2.getY()) / GRID_WIDTH)
                gb.humanRun(a2,b2)

                piece = Circle(Point(GRID_WIDTH * a2, GRID_WIDTH * b2), 16)
                piece.setFill('black')
                piece.draw(win)
                if game_win(list2):
                    message = Text(Point(100, 100), "black win.")
                    message.draw(win)
                    gb.g = 1

                change = change + 1

    message = Text(Point(100, 120), "Click anywhere to quit.")
    message.draw(win)
    win.getMouse()
    win.close()


if __name__ == "__main__":
    run()
