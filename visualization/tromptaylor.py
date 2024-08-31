import sente
def convertBoard(board) -> [[int]]:
    ans = [[0 for _ in range(board.get_side())] for _ in range(board.get_side())]
    for i in range(board.get_side()):
        for j in range(board.get_side()):
            if board.get_stone(i+1, j+1) == sente.stone.BLACK:
                ans[i][j] = 1
            elif board.get_stone(i+1, j+1) == sente.stone.WHITE:
                ans[i][j] = 2
    return ans
def tromptaylor(board: [[int]], size) -> [int, int]:
    scores = [0, 0]
    empty = set()
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                empty.add((i, j))
            elif board[i][j] == 1:
                scores[0] += 1
            elif board[i][j] == 2:
                scores[1] += 1
    while len(empty) > 0:
        count = 0
        reachBlack = False
        reachWhite = False
        queue = [empty.pop()]
        while len(queue) > 0:
            x, y = queue.pop(0)
            count+=1
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) in empty:
                        queue.append((nx, ny))
                        empty.discard((nx, ny))
                    elif board[nx][ny] == 1:
                        reachBlack = True
                    elif board[nx][ny] == 2:
                        reachWhite = True
        if reachBlack and not reachWhite:
            scores[0] += count
        elif reachWhite and not reachBlack:
            scores[1] += count
    return scores

        