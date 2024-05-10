import torch
import torch.nn as nn
import torch.optim as optim
from classes import DrumTrackerDataset, ModelTrainer
from models import Model_00


def main():
    dataset = DrumTrackerDataset()
    model = Model_00()
    loss_fn = nn.CrossEntropyLoss()
    optim_fn = optim.Adam(params=model.parameters(), lr=0.001)
    trainer = ModelTrainer(model, loss_fn, optim_fn, dataset)
    trainer.train_model(EPOCHS=10, BATCH_SIZE=64)


if __name__ == '__main__':
    main()