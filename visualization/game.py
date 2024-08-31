import sys
import sente
import random
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from visualization.algorithm import Info, sgEvaluate, adversaryEvaluate
from visualization.tromptaylor import convertBoard, tromptaylor

class Node:
    def __init__(self, move: (int, int) = None, parent = None):
        if (parent == None):
            self.turn = 0
        else:
            self.turn = parent.turn + 1
        self.parent = parent
        self.children = []
        self.showSgData = False
        self.sgData = None
        self.showAdversaryData = False
        self.adversaryData = None
        self.passes = 0
        self.resigned = False
        self.move = move;
    def needSgData(self):
        if self.sgData == None:
            moves = []
            node = self
            while node.move != None:
                moves.append(node.move)
                node = node.parent
            moves.reverse()
            self.sgData = sgEvaluate(moves, size)
    def needAdversaryData(self):
        if self.adversaryData == None:
            moves = []
            node = self
            while node.move != None:
                moves.append(["W" if node.turn % 2 == 0 else "B", str(node.move)])
                node = node.parent
            moves.reverse()
            self.adversaryData = adversaryEvaluate(moves, size)
    def toggleSgData(self):
        self.needSgData()
        self.showSgData = not self.showSgData
    def toggleAdversaryData(self):
        self.needAdversaryData()
        self.showAdversaryData = not self.showAdversaryData



class Square(QFrame):
    def __init__(self, x, y):
        super().__init__()
        self.setStyleSheet("background-color: none; color: black")
        self.hovered = False
        self.piece = "None"
        self.xcoord = x
        self.ycoord = y
        self.branchNum = -1
    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawLine(round(self.width()/2), 0, round(self.width()/2), self.height())
        painter.drawLine(0, round(self.height()/2), self.width(), round(self.height()/2))
        if self.hovered:
            if (currentNode.turn % 2 == 0):
                painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, 50)))
            painter.drawEllipse(self.rect())
        if self.piece == "Black":
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.drawEllipse(self.rect())
        elif self.piece == "White":
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(self.rect())
        if self.branchNum != -1:
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "B"+str(self.branchNum))
        if currentNode.showSgData and currentNode.sgData.pDist[self.xcoord][self.ycoord] > 0.01:
            painter.setPen(QPen(QColor(0, 200, 0)))
            painter.drawText(self.rect().adjusted(0, 0, 0, round(-1*self.height()/2)), Qt.AlignmentFlag.AlignCenter, str(round(100*currentNode.sgData.pDist[self.xcoord][self.ycoord], 2)))
        if currentNode.showAdversaryData and currentNode.adversaryData.pDist[self.xcoord][self.ycoord] > 0.01:
            painter.setPen(QPen(QColor(255, 0, 0)))
            painter.drawText(self.rect().adjusted(0, round(self.height()/2), 0, 0), Qt.AlignmentFlag.AlignCenter, str(round(100*currentNode.adversaryData.pDist[self.xcoord][self.ycoord], 2)))
    def enterEvent(self, event: QEnterEvent) -> None:
        if (game.is_legal(self.xcoord+1, self.ycoord+1) and currentNode.passes != 2 and not currentNode.resigned):
            self.hovered = True
        self.update()
    def leaveEvent(self, event: QEvent) -> None:
        self.hovered = False
        self.update()
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (game.is_legal(self.xcoord+1, self.ycoord+1) and currentNode.passes != 2 and not currentNode.resigned):
            board.playMove(self.xcoord, self.ycoord)

class PassButton(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        self.branchNum = -1
        self.sgLabel = QLabel()
        self.sgLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sgLabel.setStyleSheet("color: green;")
        self.middleLabel = QLabel()
        self.middleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.middleLabel.setStyleSheet("color: black;")
        self.adversaryLabel = QLabel()
        self.adversaryLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adversaryLabel.setStyleSheet("color: red;")
        layout = QHBoxLayout()
        layout.addWidget(self.sgLabel)
        layout.addWidget(self.middleLabel)
        layout.addWidget(self.adversaryLabel)
        self.setLayout(layout)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        global currentNode
        if currentNode.passes != 2 and not currentNode.resigned:
            if self.branchNum != -1:
                currentNode = currentNode.children[self.branchNum]
            else:
                newNode = Node("pass", currentNode)
                newNode.passes = currentNode.passes + 1
                currentNode.children.append(newNode)
                currentNode = newNode
            game.play(None)
            board.setCurrBoardState()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        self.middleLabel.setText("Pass" if self.branchNum == -1 else "Pass-B"+str(self.branchNum))
        self.sgLabel.setText(str(int(100*currentNode.sgData.passProb)) if currentNode.showSgData else "")
        self.adversaryLabel.setText(str(int(100*currentNode.adversaryData.passProb)) if currentNode.showAdversaryData else "")
class ValueBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        layout = QHBoxLayout()
        self.sgLabel = QLabel();
        self.sgLabel.setStyleSheet("color: green;")
        self.sgLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adversaryLabel = QLabel();
        self.adversaryLabel.setStyleSheet("color: red;")
        self.adversaryLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sgLabel)
        layout.addWidget(self.adversaryLabel)
        self.setLayout(layout)
    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        self.sgLabel.setText("SigmaGo Value: " + str(int(100*currentNode.sgData.value)) if currentNode.showSgData else "")
        self.adversaryLabel.setText("Adversary Value: " + str(int(100*currentNode.adversaryData.value)) if currentNode.showAdversaryData else "")
class ScoreBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        layout = QHBoxLayout()
        self.blackLabel = QLabel();
        self.blackLabel.setStyleSheet("color: black;")
        self.blackLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resultLabel = QLabel();
        self.resultLabel.setStyleSheet("color: black;")
        self.resultLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.whiteLabel = QLabel();
        self.whiteLabel.setStyleSheet("color: black;")
        self.whiteLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.blackLabel)
        layout.addWidget(self.resultLabel)
        layout.addWidget(self.whiteLabel)
        self.setLayout(layout)
    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        score = tromptaylor(convertBoard(game.get_board()), size)
        self.blackLabel.setText("Black: " + str(score[0]))
        self.whiteLabel.setText("White: " + str(score[1]))
        if currentNode.passes == 2:
            if (score[0] > score[1]):
                self.resultLabel.setText("B+"+str(score[0]-score[1]))
            elif (score[1] > score[0]):
                self.resultLabel.setText("W+"+str(score[1]-score[0]))
            else:
                self.resultLabel.setText("Draw")
        elif currentNode.resigned:
            if (currentNode.turn % 2 == 0):
                self.resultLabel.setText("W+R")
            else:
                self.resultLabel.setText("B+R")
        else:
            self.resultLabel.setText("Game in progress")
            

        
class Board(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(735, 780)
        self.setStyleSheet("background-color: #f0d9b5;")
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.squares = []
        for i in range(size):
            self.squares.append([])
            for j in range(size):
                square = Square(i, j)
                layout.addWidget(square, i, j)
                self.squares[i].append(square)
        self.passButton = PassButton()
        layout.addWidget(self.passButton, size, 0, 1, size)
        self.valueBar = ValueBar()
        layout.addWidget(self.valueBar, size+1, 0, 1, size)
        self.scoreBar = ScoreBar()
        layout.addWidget(self.scoreBar, size+2, 0, 1, size)
        self.setLayout(layout)
    def setBoardState(self, state):
        self.passButton.branchNum = -1
        for i in range(size):
            for j in range(size):
                if state.get_stone(i+1, j+1) == sente.stone.BLACK:
                    self.squares[i][j].piece = "Black"
                elif state.get_stone(i+1, j+1) == sente.stone.WHITE:
                    self.squares[i][j].piece = "White"
                else:
                    self.squares[i][j].piece = "None"
                self.squares[i][j].hovered = False
                self.squares[i][j].branchNum = -1
        for i in range(len(game.get_branches())):
            move = game.get_branches()[i];
            if (move.get_x() == size and move.get_y() == size):       
                self.passButton.branchNum = i
            else:
                self.squares[move.get_x()][move.get_y()].branchNum = i
        self.update()
    def setCurrBoardState(self):
        self.setBoardState(game.get_board())
    def playMove(self, x, y):
        global currentNode
        if self.squares[x][y].branchNum != -1:
            currentNode = currentNode.children[self.squares[x][y].branchNum]
        else:
            newNode = Node((x, y), currentNode)
            currentNode.children.append(newNode)
            currentNode = newNode
        game.play(x+1, y+1)
        self.setCurrBoardState();



class Button(QPushButton):
    def __init__(self, text):
        super().__init__(text)
    def mousePressEvent(self, event: QMouseEvent) -> None:
        global currentNode
        if (self.text() == "Resign"):
            currentNode.resigned = True
            board.setCurrBoardState()
        elif (self.text() == "Previous Turn"):
            if (currentNode.resigned):
                currentNode.resigned = False
            elif (currentNode.parent != None):
                currentNode = currentNode.parent
                game.step_up()
            board.setCurrBoardState()
        elif (self.text() == "Toggle SigmaGo Stats"):
            currentNode.toggleSgData()
            board.setCurrBoardState()
        elif (self.text() == "Toggle Adversary Stats"):
            currentNode.toggleAdversaryData()
            board.setCurrBoardState()
        elif (self.text() == "Play out 1 ply"):
            Button.playOut()
        elif (self.text() == "Play out 10 plies"):
            for _ in range(10):
                Button.playOut()
    def playOut():
        if currentNode.turn % 2 == 0:
            currentNode.needSgData()
            r = random.random()
            sum = currentNode.sgData.passProb;
            if r < sum: # pass
                board.passButton.mousePressEvent(None)
                return
            for i in range(size):
                for j in range(size):
                    sum += currentNode.sgData.pDist[i][j]
                    if r < sum:
                        board.squares[i][j].mousePressEvent(None)
                        return
        else:
            currentNode.needAdversaryData()
            r = random.random()
            sum = currentNode.adversaryData.passProb;
            if r < sum:
                board.passButton.mousePressEvent(None)
                return
            for i in range(size):
                for j in range(size):
                    sum += currentNode.adversaryData.pDist[i][j]
                    if r < sum:
                        board.squares[i][j].mousePressEvent(None)
                        return

class MainWindow(QFrame):
    instance = None
    def __init__(self):
        MainWindow.instance = self
        super().__init__()
        self.setWindowTitle("Play against SigmaGo Visualizer")
        self.setFixedSize(900, 900)

        layout = QVBoxLayout()
        controlbar = QWidget()
        controlbarLayout = QGridLayout()

        controlbarLayout.addWidget(Button("Resign"), 0, 0)
        controlbarLayout.addWidget(Button("Previous Turn"), 0, 1)
        controlbarLayout.addWidget(Button("Toggle SigmaGo Stats"), 0, 2)
        controlbarLayout.addWidget(Button("Toggle Adversary Stats"), 1, 0)
        controlbarLayout.addWidget(Button("Play out 1 ply"), 1, 1)
        controlbarLayout.addWidget(Button("Play out 10 plies"), 1, 2)
        controlbar.setLayout(controlbarLayout)

        global board
        board = Board()
        layout.addWidget(controlbar)
        layout.addWidget(board, 1)

        self.setLayout(layout)
        self.show()

        global currentNode
        currentNode = Node(None)
        global game
        game = sente.Game(size)


    

class ChooseBoardWindow(QFrame):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose Board")
        self.setFixedSize(500, 500)
        
        layout = QVBoxLayout()
        layout.addWidget(self.OptionButton("Make a 9x9 Board", 9));
        layout.addWidget(self.OptionButton("Make a 19x19 Board", 19));
    
        self.setLayout(layout)
    class OptionButton(QPushButton):
        def __init__(self, text, size):
            super().__init__(text)
            self.size = size
        def mousePressEvent(self, event: QMouseEvent) -> None:
            global size
            size = self.size
            window = MainWindow()
            window.show()
            



app = QApplication([])

window = ChooseBoardWindow();
window.show()

sys.exit(app.exec())