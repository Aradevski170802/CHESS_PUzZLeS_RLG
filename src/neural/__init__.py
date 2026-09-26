"""
PuzzleNet: a multi-task neural network that reads a chess position together with
its principal line and predicts the line's tactical category, its Lichess theme
tags, and how hard it is to find (a rating with its own uncertainty).

    encoding.py   position + line -> fixed-length feature vector
    network.py    the network, its loss, hand-derived backpropagation, AdamW
    predictor.py  the trained model as used by the analyzer, miner and app
"""
