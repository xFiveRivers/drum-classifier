import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torcheval.metrics.functional import multiclass_f1_score


class ModelTrainer():

    def __init__(self, model, loss_fn, optim_fn, scheduler, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.scheduler = scheduler
        self.dataset = dataset
        self.device = 'cuda'

        self.train_dataloader = None
        self.test_dataloader = None

        self.results = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'test_loss': [],
            'test_acc': [],
            'test_f1': []
        }


    def train_model(self, EPOCHS: int = 100, BATCH_SIZE: int = 32,
                    train_split: float = 0.7, test_split: float = 0.3,
                    rand_seed: int = 42):
        
        self._initalize_device(rand_seed)
        self._initialize_dataloaders(BATCH_SIZE, rand_seed,
                                     train_split, test_split)
        
        print(f'Beginning training with {EPOCHS} epochs...')
        
        for epoch in range(EPOCHS):
            # Get scheduler learning rate
            lr = self.scheduler.get_last_lr()[0]

            # Train epoch
            train_loss, train_acc, train_f1 = self._training_step(self.train_dataloader)

            # Evaluate epoch
            test_loss, test_acc, test_f1 = self._evaluation_step(self.test_dataloader)

            # Save epoch results
            self.results['epochs'].append(epoch + 1)
            self.results['train_loss'].append(train_loss.detach().cpu().numpy())
            self.results['train_acc'].append(train_acc.detach().cpu().numpy())
            self.results['train_f1'].append(train_f1.detach().cpu().numpy())
            self.results['test_loss'].append(test_loss.detach().cpu().numpy())
            self.results['test_acc'].append(test_acc.detach().cpu().numpy())
            self.results['test_f1'].append(test_f1.detach().cpu().numpy())

            # Print update message
            if (epoch + 1) % 1 == 0:
                update_message = (
                    f'Epoch: {epoch + 1} | '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Train F1: {train_f1:.2f} | '
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}, Test F1: {test_f1:.2f} | '
                    f'LR = {lr:.4f}'
                )
                print(update_message)
        
        # Save training results to file
        self._save_results()

        print('Training complete!')


    def _training_step(self, dataloader):
        train_loss, train_acc, train_f1 = 0, 0, 0

        self.model.train()

        for input, target in dataloader:
            input, target = input.to(self.device), target.to(self.device)

            # Get logits from model (forward pass)
            logits = self.model(input)

            # Calculate loss
            batch_loss = self.loss_fn(logits, target)
            train_loss += batch_loss

            # Gradient descent
            self.optim_fn.zero_grad()
            batch_loss.backward()
            self.optim_fn.step()

            # Calculate predictions
            preds = torch.softmax(logits, dim=1).argmax(dim=1)

            # Calculate accuracy
            batch_acc = torch.sum(preds == target) / preds.size(0)
            batch_f1 = multiclass_f1_score(preds, target, num_classes=3)
            train_acc += batch_acc
            train_f1 += batch_f1


        # Step scheduler
        self.scheduler.step()
        
        # Average out metrics
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        train_f1 = train_f1 / len(dataloader)

        return train_loss, train_acc, train_f1
    

    def _evaluation_step(self, dataloader):
        eval_loss, eval_acc, eval_f1 = 0, 0, 0

        self.model.eval()

        with torch.inference_mode():
            for input, target in dataloader:
                input, target = input.to(self.device), target.to(self.device)

                # Get logits from model (forward pass)
                logits = self.model(input)

                # Calculate loss
                batch_loss = self.loss_fn(logits, target)
                eval_loss += batch_loss

                # Calculate predictions
                preds = torch.softmax(logits, dim=1).argmax(dim=1)

                # Calculate accuracy
                batch_acc = torch.sum(preds == target) / preds.size(0)
                batch_f1 = multiclass_f1_score(preds, target, num_classes=3)
                eval_acc += batch_acc
                eval_f1 += batch_f1
        
        # Average out metrics
        eval_loss = eval_loss / len(dataloader)
        eval_acc = eval_acc / len(dataloader)
        eval_f1 = eval_f1 / len(dataloader)

        return eval_loss, eval_acc, eval_f1


    def _initalize_device(self, rand_seed):
        # Initialize device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)

        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        
        print(f'Training Device = {self.device}')
        print(f'Random Seed = {rand_seed}')


    def _initialize_dataloaders(self, BATCH_SIZE: int, rand_seed: int,
                                train_split: float, test_split: float):
        generator = torch.Generator().manual_seed(rand_seed)
        train_data, test_data = random_split(
            self.dataset, 
            [train_split, test_split], 
            generator=generator
        )

        self.train_dataloader = DataLoader(
            train_data,
            BATCH_SIZE,
            shuffle = True,
            generator = generator,
            pin_memory = True)
        
        self.test_dataloader = DataLoader(
            test_data,
            BATCH_SIZE,
            shuffle = True,
            generator = generator,
            pin_memory = True)
        
    
    def _save_results(self):
        pd.DataFrame(self.results).to_csv(
            f'results/{self.model.get_name()}_{int(time.time())}.csv',
            header = True,
            index = False
        )