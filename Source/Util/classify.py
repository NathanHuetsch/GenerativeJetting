import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from Source.Util.util import get, save_params

class ClassNN(nn.Module):
    def __init__(self, data_dim, n_layers, hidden_dim):
        super(ClassNN, self).__init__()
        layers = []
        
        layers.append(nn.Linear(data_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))  # Assuming binary classification
        layers.append(nn.Sigmoid())  # For BCELoss, we need sigmoid
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

    def batch_loss(self, input, label):
        output = self.forward(input)
        
        # Ensure the output is clipped to avoid log(0) which is undefined
        # Compute BCE loss manually
        loss = nn.BCELoss()(output, label.float())
        
        # Return the mean loss
        return loss
    
class measure_class():
    def __init__(self, x, y, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_dir = params['out_dir'] 
        self.params = params 

        self.class_samples = get(self.params, "class_samples", 10_000)
        x = x[:self.class_samples]
        y = y[:self.class_samples]

        print(f"measure_class: x shape is {x.shape}, y shape is {y.shape}")

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

        self.train = torch.cat((self.x, self.y), axis=0)
        self.labels = torch.cat((torch.zeros(self.x.shape[0]), torch.ones(self.y.shape[0])), axis=0).unsqueeze(1)
        
        self.dataset = TensorDataset(self.train, self.labels)

        # Calculate sizes for training and validation sets (90% train, 10% val)
        val_size = int(0.1 * len(self.dataset))
        train_size = len(self.dataset) - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.BATCHSIZE = get(self.params, "class_batch_size", 128)
        
        # Create DataLoaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCHSIZE, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.BATCHSIZE, shuffle=False)

        self.build_model()
        self.train_model()
        self.plot_eval()

    def build_model(self):
        self.model = ClassNN(9, 3, 32).to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"build_model: Model has {total_parameters:d} trainable parameters")

    def train_model(self):
        class_epochs = get(self.params, "class_epochs", 20)
        print("train_model: Training Class Model...")

        learning_rate = 1e-4

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(self.train_dataloader), epochs=class_epochs)
        
        def train_class_epoch(model, loader, train_epoch_losses):
            model.train()
            losses = []
            for batch, (data, label) in enumerate(loader):
                data = data.to(self.device)
                label = label.to(self.device)
                loss = model.batch_loss(data, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            train_epoch_losses.append(np.mean(losses))
            return np.mean(losses)

        def val_class_epoch(model, loader, val_epoch_loss):
            model.eval()
            losses = []
            with torch.no_grad():
                for batch, (data, label) in enumerate(loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    loss = model.batch_loss(data, label)
                    losses.append(loss.item())
            val_epoch_loss.append(np.mean(losses))
            return np.mean(losses)

        self.losses = []
        self.val_loss = []
        patience = 0
        for epoch in range(class_epochs):
            t_loss = train_class_epoch(self.model, self.train_dataloader, self.losses)
            v_loss = val_class_epoch(self.model, self.val_dataloader, self.val_loss)
            print(f"{epoch}/{class_epochs} loss: {t_loss} val_loss: {v_loss}")

            if v_loss >= t_loss:
                patience += 1
                if patience > 3:
                    print(f"train_model: Early stopping at epoch {epoch}")
                    break
            else:
                patience = 0

    def plot_eval(self):
        self.model = self.model.to('cpu')
        with torch.no_grad():
            predicted_probs = self.model.forward(self.train).cpu().numpy().flatten()
            true_labels = self.labels.cpu().numpy().flatten()

        # Plot histogram
        fig, axs = plt.subplots(1, 3, figsize=(20, 4))

        axs[0].hist(predicted_probs[true_labels == 0], range=(0, 1), density=True, bins=100, alpha=0.5, label='Data 0', color='blue')
        axs[0].hist(predicted_probs[true_labels == 1], range=(0, 1), density=True, bins=100, alpha=0.5, label='Sampled Data 1', color='red')
        axs[0].legend()
        axs[0].set_title(f'Histogram of Predicted Probabilities')
        axs[0].set_xlabel('Predicted Probability')
        axs[0].set_ylabel('Density')

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        auc_score = roc_auc_score(true_labels, predicted_probs)

        lw = 2
        axs[1].plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        axs[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.0])
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title(f'Receiver Operating Characteristic (ROC) Curve: AUC = {auc_score:.3f}')
        axs[1].legend(loc="lower right")

        # Plot loss curves
        axs[2].plot(self.losses, label='Train Loss')
        axs[2].plot(self.val_loss, label='Val Loss')
        axs[2].set_title('Loss Curve')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')
        axs[2].legend()

        self.params["AUC_SCORE"] = auc_score.item()

        print(f'AUC SCORE IS: {auc_score}')
        print(f"class_eval_mode: saved ROC plot to {self.out_dir}.pdf")
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/plots/evaluation_plot.pdf")
