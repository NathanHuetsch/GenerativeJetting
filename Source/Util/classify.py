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
        
        # Debug: Check the range of the output
        if not torch.all((output >= 0) & (output <= 1)):
            print("Output out of range:", output)
        
        # Compute BCE loss
        loss = nn.BCELoss()(output, label.float())
        
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
        samples_n = self.params.get('class_samples', 100_000)
        channels = self.params.get('plot_channels', [2, 4, 5])
        self.BATCHSIZE = self.params.get("class_batch_size", 128)

        x, y = x[:samples_n], y[:samples_n]

        # Ensure x and y are numpy arrays and not lists
        x, y = np.array(x), np.array(y)

        # Compute mass_x and mass_y
        mass_x, mass_y = get_M_ll(x), get_M_ll(y)
        print(f"mass_x shape: {mass_x.shape}, mass_y shape: {mass_y.shape}")

        # Convert to tensors and add a dimension for concatenation
        mass_x, mass_y = torch.Tensor(mass_x), torch.Tensor(mass_y)
        mass_x, mass_y = mass_x.unsqueeze(1), mass_y.unsqueeze(1)
        print(f"mass_x tensor shape: {mass_x.shape}, mass_y tensor shape: {mass_y.shape}")

        # Select the specified channels
        x, y = x[:, channels], y[:, channels]
        self.x, self.y = torch.Tensor(x), torch.Tensor(y)
        print(f"measure_class: x shape after channel selection is {self.x.shape}, y shape is {self.y.shape}")

        # Concatenate mass_x and mass_y
        self.x = torch.cat((self.x, mass_x), dim=1)
        self.y = torch.cat((self.y, mass_y), dim=1)
        self.dim = self.x.shape[1]  # Update dimension based on new feature set
        
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
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCHSIZE, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.BATCHSIZE, shuffle=False)

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
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
            print(f"{epoch}/{class_epochs} loss: {t_loss} val_loss: {v_loss}")
            
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

        with torch.no_grad():
            predicted_probs = self.model.forward(self.train).cpu().numpy().flatten()
            true_labels = self.labels.cpu().numpy().flatten()

        # Create a PDF file to save plots
        pdf_path = f"{self.out_dir}/plots/evaluation_plots.pdf"
        with PdfPages(pdf_path) as pdf:
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

            bins = np.linspace(0, 1, 60)  # Define the bins for the histogram
            range_ = (0, 1)  # Define the range for the histogram

            fig1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios" : [3, 1], "hspace" : 0.00})
            fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

            # Calculate histograms
            y_t, _ = np.histogram(predicted_probs[true_labels == 0], bins=bins, range=range_)
            y_tr, _ = np.histogram(predicted_probs[true_labels == 1], bins=bins, range=range_)
            y_g, _ = np.histogram(predicted_probs, bins=bins, weights=(true_labels == 1).astype(float))

            hists = [y_t, y_g, y_tr]
            hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]

            integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
            scales = [1 / integral if integral != 0. else 1. for integral in integrals]

            FONTSIZE = 16
            labels = ['True Data', 'Predicted Data', 'Sampled Data']
            colors = ["#0343DE", "#A52A2A", "black"]
            dup_last = lambda a: np.append(a, a[-1])

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
            roc_data_path = f"{self.out_dir}/plots/roc_data.npz"
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

            print(f'pred. probs shape: {predicted_probs.shape}')
            print(f'weight shape {weights.shape}')

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

        # Ranges of the observables for plotting
            self.obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [17,  157], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50],
                           [17,  82], [-4, 4], [-6, 6], [0, 50]]

            channels = get(self.params, "channels", None)
            if channels is None:
                channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()

            # Plot histograms for the first 20 dimensions
            for i, channel in enumerate(channels):                
                fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios" : [3, 1], "hspace" : 0.00})
                fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

                bins = np.linspace(self.x[:, i].min(), self.x[:, i].max(), 60)
                y_t, _ = np.histogram(self.x[:, i], bins=bins, range= self.obs_ranges[channel])
                y_g, _ = np.histogram(self.y[:, i], bins=bins, weights=weights) if weights.size > 0 else np.histogram(self.y[:, i], bins=bins,range= self.obs_ranges[channel])
                y_tr, _ = np.histogram(self.y[:, i], bins=bins,range= self.obs_ranges[channel])

                hists = [y_t, y_g, y_tr]
                hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]

                integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
                scales = [1 / integral if integral != 0. else 1. for integral in integrals]

                FONTSIZE = 16
                labels = ['Data', 'reweighted samples', 'Sampled data']
                colors = ["black","#0343DE", "#A52A2A", ]
                dup_last = lambda a: np.append(a, a[-1])

                for y, y_err, scale, label, color in zip(hists, hist_errors, scales, labels, colors):
                    axs[0].step(bins, dup_last(y) * scale, label=label, color=color, linewidth=1.0, where="post")
                    axs[0].step(bins, dup_last(y + y_err) * scale, color=color, alpha=0.5, linewidth=0.5, where="post")
                    axs[0].step(bins, dup_last(y - y_err) * scale, color=color, alpha=0.5, linewidth=0.5, where="post")
                    axs[0].fill_between(bins, dup_last(y - y_err) * scale, dup_last(y + y_err) * scale, facecolor=color, alpha=0.3, step="post")

                    if label == "Original": continue

                    ratio = (y * scale) / (hists[2] * scales[2])
                    ratio_err = np.sqrt((y_err / y)**2 + (hist_errors[2] / hists[2])**2)
                    ratio_isnan = np.isnan(ratio)
                    ratio[ratio_isnan] = 1.
                    ratio_err[ratio_isnan] = 0.

                    axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
                    axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5, linewidth=0.5, where="post")
                    axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5, linewidth=0.5, where="post")
                    axs[1].fill_between(bins, dup_last(ratio - ratio_err), dup_last(ratio + ratio_err), facecolor=color, alpha=0.3, step="post")

                axs[0].legend(loc="center right", frameon=False, fontsize=FONTSIZE)
                axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
                axs[1].set_ylabel(r"$\frac{\mathrm{Generated}}{\mathrm{Original}}$", fontsize=FONTSIZE)
                axs[1].set_ylim(0.71, 1.29)
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                plt.xlabel(f"dimension {i}", fontsize=FONTSIZE)

                axs[0].tick_params(axis="both", labelsize=FONTSIZE)
                axs[1].tick_params(axis="both", labelsize=FONTSIZE)
                pdf.savefig(fig)  # Save the histogram for each dimension to the PDF
                plt.close(fig)

