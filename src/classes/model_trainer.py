import torch

class ModelTrainer():

    def __init__(self, model, loss_fn, EPOCHS: int = 100, LR: float = 0.01,):
        self. model = model
        self.loss_fn = loss_fn
        self.EPOCHS = EPOCHS
        self.LR = LR


    def train(self):
        pass


    def _evaluate(self):
        pass