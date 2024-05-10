import torch
import torch.nn as nn
import torch.optim as optim
from classes import DrumTrackerDataset, CNN_Model, ModelTrainer


def main():
    dataset = DrumTrackerDataset()
    cnn_model = CNN_Model()
    loss_fn = nn.CrossEntropyLoss()
    optim_fn = optim.Adam(params=cnn_model.parameters(), lr=0.001)
    trainer = ModelTrainer(cnn_model, loss_fn, optim_fn, dataset)
    trainer.train_model(EPOCHS=10, BATCH_SIZE=64)


if __name__ == '__main__':
    main()