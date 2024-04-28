import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class ModelTrainer():

    def __init__(self, model, loss_fn, optim_fn, dataset,
                 EPOCHS: int = 100, BATCH_SIZE: int = 32,
                 rand_seed: int = 42, train_split: float = 0.7,
                 test_split: float = 0.3):
        self.model = model
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.dataset = dataset
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE,
        self.rand_seed = rand_seed
        self.train_split = train_split
        self.test_split = test_split
        self.device = 'cuda'

        self.train_dataloader = None
        self.test_dataloader = None

        self.epoch_counts = []
        self.train_losses = []
        self.test_losses = []


    def train_model(self):
        self._initalize_device()
        self._initialize_dataloaders()
        print(f'Beginning training with {self.EPOCHS} epochs...')
        
        for epoch in range(self.EPOCHS):
            # Set model to training mode
            self.model.train()
            train_loss, train_acc = self._training(self.train_dataloader)

            # Gradient descent
            self.optim_fn.zero_grad()
            train_loss.backward()
            self.optim_fn.step()

            # Set model to evaluation mode
            self.model.eval()
            with torch.inference_mode():
                test_loss, test_acc = self._training(self.test_dataloader)

            if epoch % 5 == 0:
                self.epoch_counts.append(epoch + 1)
                self.train_losses.append(train_loss.detach().numpy())
                self.test_losses.apppend(test_loss.detach().numpy())
                print(f'Epoch: {epoch} | \
                    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} | \
                    Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
                
        print('Training complete!')


    def _training(self, dataloader):
        for input, target in dataloader:
            input, target = input.to(self.device), target.to(self.device)

            # Get  logits, predictions, and loss
            logits = self.model(input)
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            loss = self.loss_fn(logits, target)
            acc = torch.sum(preds == target) / preds.size(0)

        return loss, acc


    def _initalize_device(self):
        # Initialize device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if self.rand_seed:
            torch.manual_seed(self.rand_seed)
            torch.cuda.manual_seed(self.rand_seed)
        
        print(f'Training Device = {self.device}')
        print(f'Random seed = {self.rand_seed}')


    def _initialize_dataloaders(self):
        print('Splitting dataset...')
        generator = torch.Generator().manual_seed(42)
        train_data, test_data = random_split(
            self.dataset, 
            [self.train_split, self.test_split], 
            generator=generator
        )

        print('Initializing dataloader for training...')
        self.train_dataloader = DataLoader(
            train_data,
            self.BATCH_SIZE,
            generator=generator)
        
        print('Initializing dataloader for testing...')
        self.test_dataloader = DataLoader(
            test_data,
            self.BATCH_SIZE,
            generator=generator)