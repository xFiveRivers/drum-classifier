"""
Model Training Script

Usage:
    train_model.py --epochs=<epochs> --lr=<lr> --batch=<batch>

Options:
    -h --help                   Show help screen.
    --epochs=<epochs>           Number of epochs to train for [default: 10]
    --lr=<lr>                   Learning rate for optimizer [default: 0.01]
    --batch=<batch>             Samples per training batch [default: 32]
"""


import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from classes import DrumTrackerDataset, ModelTrainer
from docopt import docopt
from models import Model_00, Model_01


# Parse args from CLI
args = docopt(__doc__)


def main(epochs, lr, batch):
    # Instatiate classes
    dataset = DrumTrackerDataset()
    model = Model_01()

    # Get loss and optimizer functions
    loss_fn = nn.CrossEntropyLoss()
    optim_fn = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = lr_scheduler.ExponentialLR(optim_fn, gamma=0.95)

    # Train model
    trainer = ModelTrainer(model, loss_fn, optim_fn, scheduler, dataset)
    trainer.train_model(EPOCHS=epochs, BATCH_SIZE=batch)


if __name__ == '__main__':
    main(int(args['--epochs']), float(args['--lr']), int(args['--batch']))