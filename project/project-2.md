---
tags:
  - notes
comments: true
dg-publish: true
---

# MultiAgent

> [!PREREQUISITE]
>
> - [05-Trees_Minimax_Pruning](../note/05-Trees_Minimax_Pruning.md)
> - [06-Expectimax_Monte_Carlo_Tree_Search](../note/06-Expectimax_Monte_Carlo_Tree_Search.md)
> - [preject 2](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj2/) （若需要认证，可见[仓库](https://github.com/Darstib/cs188/tree/main/materials/project/intro_page)）

## Quick Review

- 博弈树（Game Tree）
	- 博弈树是一种图形结构，用于表示博弈中的所有可能状态及其相应的决策。树的节点代表游戏状态，边代表玩家的行动。通过分析博弈树，玩家可以评估不同策略的结果，从而选择最佳行动。
- Minimax（极小极大算法）
	- 一种决策算法，用于在零和游戏中寻找最佳策略，通过假设对手也会采取最佳行动，从而最小化可能的损失或最大化可能的收益。
- Alpha-Beta Pruning（α-β 剪枝）
	- 一种优化算法，用于减少在博弈树中评估的节点数量，从而提高 minimax 算法的效率，常用于决策过程中。
- Evaluation Functions（评估函数）
	- 在决策中用于估计特定局面的函数，帮助 Agent 判断当前状态的优劣，以便做出最佳决策。
- Expectimax（期望极大算法）
	- 一种扩展的决策算法，适用于包含随机性因素的博弈，通过计算每个可能结果的期望值来选择最佳行动，常用于棋类游戏和其他不确定环境中的决策。
- Mixed Layer Types（混合层类型）
	- 在决策过程中，我们的对手不一定只有一个（但是目的相同），他们依次行动，每人依次在博弈树上占有一层。
- General Game（通用游戏）
	- 我们的多个对手的目的并非相同，对手的执行顺序对结果影响巨大。
- Monte Carlo Tree Search（蒙特卡洛树搜索）
	- 蒙特卡洛树搜索是一种启发式搜索算法，通过随机采样和模拟来评估游戏树中的节点，帮助做出决策。

## explain

### Q1 (4 pts): Reflex Agent

文档提醒我们关注：

```python title="game.py -> Grid"
class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal, y vertical and the origin (0,0) in the bottom left corner.
    """
    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]:
            raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30
    
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(
            height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)
    ... # 省略一些关系不大的内容
    def asList(self, key=True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key:
                    list.append((x, y))
        return list
```

#### explore

对于 evalution，我们之前使用过曼哈顿距离，这里不妨继续使用；不同的是，这次我们应该把 ghost 的位置也考虑进来了；此外，距离食物越近，鬼魂越远，得分应当越高，这里按照 project 2 中的提示使用倒数：

```python title="evalutionFunction v1"
def evaluationFunction(self, currentGameState: GameState, action):
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # return successorGameState.getScore()
    "*** YOUR CODE HERE ***"
    # Get the food list and initial score
    foodList = newFood.asList()
    score = successorGameState.getScore()
    # Fooe evaluation
    for food in foodList:
        score += 1 / util.manhattanDistance(newPos, food)
    
    # Ghost evaluation
    for i, ghostState in enumerate(newGhostStates):
        ghostPos = ghostState.getPosition()
        distance = util.manhattanDistance(newPos, ghostPos)
        if newScaredTimes[i] > 0:
            # Ghost is scared, it's good to be close
            score += 2 / (distance + 1) # avoid ZeroDivisionError
        else:
            score -= 2 / (distance + 1)  
    
    return score
```

[Q1 半通过](attachments/project-2-1.png)

#### right

可以看到死了两次，但是平均分还是挺高的；但是依照我们的成绩标准，不死才是比较好的；所以我们离 ghost 远一点，近了就给个“差评”：

```python title="evalutionFunciotn v2"
def evaluationFunction(self, currentGameState: GameState, action):
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # return successorGameState.getScore()
    "*** YOUR CODE HERE ***"
    # Get the food list and initial score
    foodList = newFood.asList()
    score = successorGameState.getScore()
    for food in foodList:
        score += 1 / util.manhattanDistance(newPos, food)
    
    # Ghost evaluation
    for i, ghostState in enumerate(newGhostStates):
        ghostPos = ghostState.getPosition()
        distance = util.manhattanDistance(newPos, ghostPos)
        if newScaredTimes[i] > 0:
            # Ghost is scared, it's good to be close
            score += 2 / (distance + 1) # avoid ZeroDivisionError
        else:
            if distance < 2: # min_distance
                score -= 500  # Big penalty for being too close
            else:
                score -= 2 / (distance + 1)
    
    return score
```

[Q1 通过](attachments/project-2.png)

可以看到最后一次为了躲避 ghost 分数低于了 1000，但是活着才是硬道理。当然存在随机性，[都高于 1000](attachments/project-2-3.png) 也不是不可；如果将 min_distance 改为 1，效果和 v1 差不多，可能是逃跑的太晚了吧 hhh 。

最后再来测试一下 `python pacman.py --frameTime 0 -p ReflexAgent -k 2` 

![](attachments/project-2-2.png)

可以看到也成功通过了；min_distance = 1 时则失败。

### Q2 (5 pts): Minimax

#### explore

在 project-2 中给出了一个 depth=2 时的决策树，我将其分为若干层 level （最上方为 level=0）：

![概念图](attachments/project-2-4.png)

思路：getAction 仿照 Q1 中的

```python title="MinimaxAgent -> getAction v1"
  def getAction(self, gameState: GameState):
        """
        使用 self.depth 和 self.evaluationFunction 从当前 gameState 返回极大极小动作。
        这里有一些在实现极大极小时可能有用的方法调用。
        gameState.getLegalActions(agentIndex):
        返回一个代理的合法动作列表
        agentIndex=0 表示 Pacman 幽灵的索引 >= 1
        gameState.generateSuccessor(agentIndex, action):
        返回代理执行动作后的后继游戏状态
        gameState.getNumAgents():
        返回游戏中的代理总数
        gameState.isWin():
        返回游戏状态是否为胜利状态
        gameState.isLose():
        返回游戏状态是否为失败状态
        """
        "*** 你可以在这里编写你的代码 ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:#取max
                return max(minimax(1, depth, gameState.generateSuccessor(0, action)) for action in gameState.getLegalActions(0))
                # return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
            else:
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextid = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth
                if nextAgent == 0: #下一个是max ，深度加一
                    nextDepth = depth + 1
                # nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextid, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
        legalMoves = gameState.getLegalActions(0)
        # 选择一个最佳的动作
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # 在最佳动作中随机选择一个
        return legalMoves[chosenIndex]
        util.raiseNotDefined()

```

[Q2 通过](attachments/project-2-5.png)



### Q3 (5 pts): Alpha-Beta Pruning

#### explore

在 [05-Trees_Minimax_Pruning](../note/05-Trees_Minimax_Pruning.md) 中我们提到了 [Alpha-Beta Pruning 的伪代码](attachments/project-2-10.png)，由上一题的代码，我们将其 agentIndex 不同时的操作更换即可：

#### right

参考 [szzxljr 的代码](https://github.com/szzxljr/CS188_Course_Projects/blob/master/proj2multiagent/multiAgents.py#L191) 我发现了问题：我在最后获取根节点的值是依旧遍历了其所有子代；在最后我们依旧应该剪枝：

```python title="AlphaBetaAgent -> getAction v2"
    def getAction(self, gameState: GameState):
        """
        使用 self.depth 和 self.evaluationFunction 返回带有 alpha-beta 剪枝的极大极小动作
        """
        "*** 你可以在这里编写你的代码 ***"
        def alpha_beta(agentIndex, depth, gameState, alpha, beta): #alpha和beta是值传递，相当于只有兄弟节点会互相影响
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman's turn (maximizing player)
                value = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    value = max(value, alpha_beta(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (minimizing player)
                value = float('inf')
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    value = min(value, alpha_beta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        legalMoves = gameState.getLegalActions(0)
        # 选择一个最佳的动作
        a = -float('inf')
        b = float('inf')
        bestscore = -float('inf')
        bestaction = None
        for action in legalMoves:
            scores = alpha_beta(1, 0, gameState.generateSuccessor(0, action),a , b) 
            if scores > bestscore:
                bestscore = scores
                bestaction = action
            a = max(a, bestscore)
        
        return bestaction
```

[Q3 通过](attachments/project-2-9.png)

### Q4 (5 pts): Expectimax

#### right

为什么 Q4 没有 explore? expectimax 和 minimax 的区别只有一个，ghost level 的效果不再是取最小，而是取平均，改改 Q2 不就好了：

```python title="ExpectimaxAgent -> getAction"
def getAction(self, gameState: GameState):
    "*** YOUR CODE HERE ***"
    def getValue(state, agentIndex, depth):
        agentIndex = agentIndex % state.getNumAgents()
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        elif agentIndex == 0:
            return max(
                getValue(
                    state.generateSuccessor(agentIndex, action),
                    agentIndex + 1,
                    depth - 1,
                )
                for action in state.getLegalActions(agentIndex)
            )
        else: # 唯一改动之处
            return sum(
                getValue(
                    state.generateSuccessor(agentIndex, action),
                    agentIndex + 1,
                    depth - 1,
                )
                for action in state.getLegalActions(agentIndex)
            ) / len(state.getLegalActions(agentIndex))
    
    # Pacman is always agent 0, and the agents move in order of increasing agent index.
    legalActions = gameState.getLegalActions(0)
    scores = [
        getValue(
            gameState.generateSuccessor(0, action),
            1,
            self.depth * gameState.getNumAgents() - 1,
        )
        for action in legalActions
    ]
    bestScore = max(scores)
    bestIndices = [
        index for index in range(len(scores)) if scores[index] == bestScore
    ]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalActions[chosenIndex]
    util.raiseNotDefined()
```

[Q4 通过](attachments/project-2-6.png)

![](attachments/project-2-8.png)

[`ExpectimaxAgent` wins about half the time](attachments/project-2-7.png), while [`AlphaBetaAgent` always loses](attachments/project-2-11.png).

### Q5 (6 pts): Evaluation Function

#### right

其实和 Q1 差不多（因为我们当时的实现就挺不错了），改为评估当前状态就好了。

```python title="betterEvaluationFunction"
def betterEvaluationFunction(currentGameState: GameState):
    """
    DESCRIPTION: <write something here so we know what you did>
    Just as what we do in ReflexAgent, but now we evaluate currentGameState
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    foodList = Food.asList()
    score = currentGameState.getScore()
    for food in foodList:
        score += 1 / util.manhattanDistance(Pos, food)
    # Ghost evaluation
    for i, ghostState in enumerate(GhostStates):
        ghostPos = ghostState.getPosition()
        distance = util.manhattanDistance(Pos, ghostPos)
        if ScaredTimes[i] > 0:
            # Ghost is scared, it's good to be close
            score += 2 / (distance + 1)
        else:
            # Ghost is not scared, avoid it
            if distance < 2:
                score -= 500  # Big penalty for being too close
            else:
                score -= 2 / (distance + 1)
    return score
    util.raiseNotDefined()
```

[Q5 通过](attachments/project-2-12.png)

## pass

- [project-2 全部通过](attachments/project-2-13.png)
- [全代码](https://github.com/Darstib/cs188/tree/main/project/solution)
