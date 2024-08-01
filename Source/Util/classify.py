import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from Source.Util.util import get, save_params
from Source.Util.physics import get_M_ll
from matplotlib.backends.backend_pdf import PdfPages
from Source.Util.plots import plot_obs, delta_r, plot_deta_dphi, plot_obs_2d, plot_loss
import os 

class ClassNN(nn.Module):
    def __init__(self, data_dim):
        super(ClassNN, self).__init__()

        layers = []
        layers.append(nn.Linear(data_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 128))        
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))   
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 16))   
        layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 1))  
        layers.append(nn.Sigmoid())        
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

    def batch_loss(self, input, label):
        output = self.forward(input)
        c = torch.isnan(output)
        i = torch.isnan(input)
        a = output > 1 
        b = output < 0

        try:
            loss = nn.BCELoss()(output, label.float())
        except: 
            print(f"error{output.min()},{output.max()}")
            print(i.sum())
            print(c.sum())
            print(a.sum())
            print(b.sum())

        if torch.isnan(output).sum() > 0:
            print(f"error{output.min()},{output.max()}")
            print(i.sum())
            print(c.sum())
            print(a.sum())
            print(b.sum())
        return loss
    
class MeasureClass:
    def __init__(self, x, y, params, label):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_dir = params['out_dir']
        self.params = params
        self.label = label
        self.dim = 9  # Default dimension without mass_x and mass_y

        self.prepare_data(x, y)
        self.build_model()
        self.train_model()
        self.plot_eval()

    def prepare_data(self, x, y):
        samples_n = self.params.get('n_samples', 100_000)
        channels = self.params.get('plot_channels', [2, 4, 5])

        self.BATCHSIZE = self.params.get("class_batch_size", 128)

        self.mass_x, self.mass_y = get_M_ll(x), get_M_ll(y)

        print(f"measure_class: mass_x shape: {self.mass_x.shape}, mass_y shape: {self.mass_y.shape}")

        # Convert to tensors and add a dimension for concatenation
        mass_x, mass_y = torch.Tensor(self.mass_x), torch.Tensor(self.mass_y)
        mass_x, mass_y = mass_x.unsqueeze(1), mass_y.unsqueeze(1)
        a = torch.isnan(mass_x).squeeze(1)
        b = torch.isnan(mass_y).squeeze(1)
        
        print(a.sum())
        print(b.sum())
        print(f'mass_x min{mass_x.min()} max:{mass_x.max()} mass_y min{mass_y.min()} max:{mass_y.max()}')
        print(f"measure_class: mass_x tensor shape: {mass_x.shape}, mass_y tensor shape: {mass_y.shape}")

        mass_x = mass_x[~a]
        x = x[~a]

        mass_x = mass_x[:samples_n]
        x = x[:samples_n]

        # Select the specified channels
        x, y = x[:, channels], y[:, channels]
        self.x, self.y = torch.Tensor(x), torch.Tensor(y)
        print(f"measure_class: x shape after channel selection is {self.x.shape}, y shape is {self.y.shape}")


        # Concatenate mass_x and mass_y
        
        self.x = torch.cat((self.x, mass_x), dim=1)
        self.y = torch.cat((self.y, mass_y), dim=1)
        self.dim = 10  
        
        print(f"measure_class: final x shape is {self.x.shape}, y shape is {self.y.shape}")
        self.train = torch.cat((self.x, self.y), axis=0).to(self.device)
        print(f"measure_class: train shape is {self.train.shape}")

        self.labels = torch.cat((torch.zeros(self.x.shape[0]), torch.ones(self.y.shape[0])), axis=0).unsqueeze(1).to(self.device)
        print(f"measure_class: labels shape is {self.labels.shape}")

        self.dataset = TensorDataset(self.train, self.labels)
        
        # Split dataset into training and validation sets
        val_split = 0.2  # Define the validation split
        val_size = int(val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.BATCHSIZE, shuffle=True)
        #self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.BATCHSIZE, shuffle=False)

    def build_model(self):
        self.model = ClassNN(self.dim).to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"build_model: Model has {total_parameters:d} trainable parameters")

    def train_model(self):
        class_epochs = get(self.params, "class_epochs", 20)
        print(f"train_model: Training Class Model on Data and {self.label}...")

        LEARNING_RATE = 1e-4

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(self.train_dataloader), epochs=class_epochs)
        
        def train_class_epoch(model, loader, train_epoch_losses):
            model.train()
            losses = []
            for batch, (data, label) in enumerate(loader):
                data = data.to(self.device)
                label = label.to(self.device)
                loss = model.batch_loss(data, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                optimizer.zero_grad()
            train_epoch_losses.append(np.mean(losses))
            return np.mean(losses)

        def val_class_epoch(model, loader, val_epoch_losses):
            model.eval()
            losses = []
            with torch.no_grad():
                for batch, (data, label) in enumerate(loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    loss = model.batch_loss(data, label)
                    losses.append(loss.item())
            val_epoch_losses.append(np.mean(losses))
            return np.mean(losses)

        self.losses = []
        self.val_loss = []
        patience = 0
        for epoch in range(class_epochs):
            t_loss = train_class_epoch(self.model, self.train_dataloader, self.losses)
            v_loss = val_class_epoch(self.model, self.val_dataloader, self.val_loss)
            print(f"{epoch}/{class_epochs} loss: {t_loss} val_loss: ")
            
            if v_loss > t_loss:
                patience += 1
                if patience > 10:
                    print(f"train_model: Early stopping at epoch {epoch}")
                    break
            else:
                patience = 0
            
                
    def plot_eval(self):
        self.model = self.model.to('cpu')
        self.train = self.train.to('cpu')
        self.labels = self.labels.to('cpu')
        self.x = self.x.cpu()
        self.y = self.y.cpu()

        self.obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                          "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                          "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                          "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}",
                          "p_{T,j3}", "\phi_{j3}", "\eta_{j3}", "\mu_{j3}"]

        self.obs_units = ["GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV",
                          "GeV", None, None, "GeV"]

        self.obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [17,  157], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50]]

        with torch.no_grad():
            predicted_probs = self.model.forward(self.train).cpu().numpy().flatten()
            true_labels = self.labels.cpu().numpy().flatten()

        pdf_path = f"{self.out_dir}/evaluation_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            # Plot Histrograms
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(predicted_probs[true_labels == 0], range=(0, 1), density=True, bins=100, alpha=0.4, label='Data 0', color='blue')
            ax1.hist(predicted_probs[true_labels == 1], range=(0, 1), density=True, bins=100, alpha=0.4, label='Sampled Data 1', color='red')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.set_xlim([0.0, 1.0])
            ax1.set_title(f'{self.label} Histogram of Predicted Probabilities')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Density')
            pdf.savefig(fig1)  # Save the histogram to the PDF
            plt.close(fig1)


            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
            roc_auc = auc(fpr, tpr)
            auc_score = roc_auc_score(true_labels, predicted_probs)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            lw = 2
            ax2.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.0])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title(f'{self.label} Receiver Operating Characteristic (ROC) Curve: AUC = {auc_score:.3f}')
            ax2.legend(loc="lower right")
            pdf.savefig(fig2)  # Save the ROC curve to the PDF
            plt.close(fig2)

            # Save ROC data to a file
            roc_data_path = f"{self.out_dir}/roc_data.npz"
            np.savez(roc_data_path, fpr=fpr, tpr=tpr, thresholds=thresholds)
            print(f"ROC data saved to {roc_data_path}")
            
            # Plot loss curves
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(self.losses, label='Train Loss')
            ax3.plot(self.val_loss, label='Val Loss')
            ax3.set_title(f'{self.label} Loss Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            pdf.savefig(fig3)  # Save the loss curves to the PDF
            plt.close(fig3)

            # Reweighting based on predicted probabilities
            weights = predicted_probs / (1 - predicted_probs)
            weights = weights[true_labels == 1]

                        
                        
            channels = get(self.params, "channels", None)
            if channels is None:
                channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()

            # Plot histograms for the first 20 dimensions
            for i, channel in enumerate(channels):
                obs_train = self.x[:,i]
                obs_test = self.y[:,i]
                obs_generated = self.y[:,i]
                obs_name = self.obs_names[channel]
                obs_range = self.obs_ranges[channel]
                label = ["REWEIGHTGEN GEN", "GEN", "DATA"]
                
                # Create the plot
                plot_obs(pp=pdf,
                            obs_train=obs_train,
                            obs_test=obs_test,
                            obs_predict=obs_generated,
                            name=obs_name,
                            range=obs_range,
                            n_epochs=0,
                            n_jets=1,
                            weight_samples=1,
                            predict_weights=weights,
                            lab = label)


            obs_name = "M_{\ell \ell}"
            obs_range = [75,110]
            bin_num = 40
            data_train = self.mass_x
            data_test = self.mass_y
            data_generated = self.mass_y

            plot_obs(pp=pdf,
                        obs_train=data_train,
                        obs_test=data_test,
                        obs_predict=data_generated,
                        name=obs_name,
                        n_epochs=0,
                        range=obs_range,
                        n_jets=1,
                        weight_samples=1,
                        predict_weights=weights,
                        lab = label)  