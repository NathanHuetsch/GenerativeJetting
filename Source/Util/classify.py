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
    def __init__(self, n_layers=5, dim_in=10, n_hidden=128, dropout=0.1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        layers = []
        layers.append(nn.Linear(dim_in, n_hidden))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, input):
        input = input.to(self.device)
        output = self.net(input)
        return output

    def batch_loss(self, data):
        input, label = data
        input = input.to(self.device)
        label = label.to(self.device)
        output = self.forward(input)

        loss_fn = nn.BCEWithLogitsLoss()

        loss = loss_fn(output, label)
        """
        c = torch.isnan(output)
        i = torch.isnan(input)
        a = output > 1 
        b = output < 0
        try:
            
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
        """
        return loss
    
class MeasureClass:
    def __init__(self, x, y, params, label):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_dir = params['out_dir']
        self.params = params
        self.label = label # Default dimension without mass_x and mass_y

        self.prepare_data(x, y)
        self.build_model()
        self.train_model()
        self.plot_eval()

    def prepare_data(self, x, y):
        samples_n = self.params.get('n_samples', 100_000)
        channels = self.params.get('plot_channels', [2, 4, 5])
        self.BATCHSIZE = self.params.get("class_batch_size", 128)

        def add_mass_to_data(data):
            mass = get_M_ll(data)

            mass = torch.Tensor(mass)
            data = torch.Tensor(data)

            mass = mass.unsqueeze(1)

            corupt_mass = torch.isnan(mass).squeeze(1)
            mass = mass[~corupt_mass]
            data = data[~corupt_mass]

            mass = mass[:samples_n]
            data = data[:samples_n]

            data = data[:, channels]

            data = torch.cat((data, mass), dim=1)
            print(f'shape after adding mass to data: {data.shape}')
            return data.numpy()

        self.x = add_mass_to_data(x)
        self.y = add_mass_to_data(y)

 
        self.data_real = self.x
        self.data_gen = self.y
        print(self.data_real.shape, self.data_gen.shape)

        self.data = np.concatenate((self.data_real, self.data_gen), axis=0)
        self.labels = np.concatenate((np.zeros(self.data_real.shape[0]), np.ones(self.data_gen.shape[0])), axis=0)
        idx = np.random.permutation(len(self.data))
        self.data = self.data[idx,:]
        self.labels=self.labels[idx,None]

        def preprocess(event, mean=None, std=None):
            if mean is None or std is None:
                mean = event.mean(axis=0, keepdims=True)
                std = event.std(axis=0, keepdims=True)
            event = (event - mean) / std
            return event, mean, std

        def create_dataloader(data, labels, batchsize, shuffle, mean=None, std=None):
            data, mean, std = preprocess(data, mean, std)
            data = torch.tensor(data).float()
            labels = torch.tensor(labels)
            loader = DataLoader(TensorDataset(data, labels), batch_size=batchsize, shuffle=shuffle)
            return loader, mean, std
        
        total_data_points = self.data.shape[0]
        n1 = int(0.6 * total_data_points) 
        n_val = int(0.5 * (total_data_points - n1))  # 50% of the remaining 40% for validation
        n2 = n1 + n_val

        batchsize = self.BATCHSIZE
        self.data_trn, self.data_val, self.data_tst = self.data[:n1,:], self.data[n1:n2,:], self.data[n2:,:]
        self.labels_trn, self.labels_val, self.labels_tst = self.labels[:n1,:], self.labels[n1:n2,:], self.labels[n2:,:]

        mean, std = None, None
        self.loader_trn, mean, std = create_dataloader(self.data_trn, self.labels_trn, batchsize, True, mean=mean, std=std)
        self.loader_tst, mean, std = create_dataloader(self.data_tst, self.labels_tst, batchsize, False, mean=mean, std=std)
        self.loader_val, mean, std = create_dataloader(self.data_val, self.labels_val, batchsize, False, mean=mean, std=std)

    def build_model(self):
        self.model = ClassNN().to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"build_model: Model has {total_parameters:d} trainable parameters")

    def train_model(self):
        class_epochs = get(self.params, "class_epochs", 20)
        print(f"train_model: Training Class Model on Data and {self.label}...")

        LEARNING_RATE = 1e-4

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,len(self.loader_trn))
        
        def train_epoch(loader, losses):
            self.model.train()
            for data in loader:
                loss = self.model.batch_loss(data)                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                losses.append(loss.item())

        def val_epoch(loader):
            losses = []
            self.model.eval()
            with torch.no_grad():
                for data in loader:
                    loss = self.model.batch_loss(data)
                    losses.append(loss.item())
            return np.mean(losses)

        self.losses = []
        self.val_loss = []

        for epoch in range(class_epochs):
            train_epoch(self.loader_trn, self.losses)

            val_loss = val_epoch(self.loader_val)
            self.val_loss.append(val_loss)

            print(f"{epoch}/{class_epochs} val_loss:{val_loss:0.5f}")
    
    def plot_eval(self):
        truth, pred = [], []

        self.model.eval()
        with torch.no_grad():
            for (x, y) in self.loader_tst:
                x = x.to('cpu')
                y = y.to('cpu')
                y_pred = self.model(x).cpu()
                truth.append(y.flatten().numpy())
                pred.append(y_pred.flatten().numpy())
        pred = np.concatenate(pred)
        truth = np.concatenate(truth)
        print(pred.shape, truth.shape)

        

        pred_sig = 1/ (1+np.exp(-pred))

        pdf_path = f"{self.out_dir}/evaluation_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            # Plot Histrograms
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(pred_sig[truth == 0], range=(0, 1), bins=100, alpha=0.4, label='Data 0', color='blue') #density=True
            ax1.hist(pred_sig[truth == 1], range=(0, 1), bins=100, alpha=0.4, label='Sampled Data 1', color='red')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.set_xlim([0.0, 1.0])
            ax1.set_title(f'{self.label} Histogram of predicted events')
            ax1.set_xlabel('classifier score')
            ax1.set_ylabel('Events')
            pdf.savefig(fig1)  # Save the histogram to the PDF
            plt.close(fig1)


            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(truth, pred)
            roc_auc = auc(fpr, tpr)
            auc_score = roc_auc_score(truth, pred)

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
            ax3.plot(np.arange(len(self.losses)),self.losses, label='Train Loss')
            ax3.plot(np.arange(len(self.val_loss))*len(self.loader_trn),self.val_loss, label='Vall Loss')
            ax3.set_title(f'{self.label} Loss Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            pdf.savefig(fig3)  # Save the loss curves to the PDF
            plt.close(fig3)

            weights = np.exp(pred)
            print(f"all weights shape {weights.shape}")
            #weights_GEN_to_DATA = weights[truth==1]
            # Reweighting based on predicted probabilities
            #weights = predicted_probs / (1 - predicted_probs)
            #weights=weights[truth == 1]

            print(f"all weights after label cut {weights.shape}")
            print(self.y.shape)
            

            test_REAL = self.data_tst[self.labels_tst[:,0]==0]
            test_GEN = self.data_tst[self.labels_tst[:,0]==1]
            print(f"test_REAL shape {test_REAL.shape}")
            print(f"test_GEN shape {test_GEN.shape}")



            weights_GENtoDATA = weights[self.labels_tst[:,0]==1]
            print(f"WEIGHTS shape {weights_GENtoDATA.shape}")
            
            """
            def plot_hist(ax, data, label, color, bins=40, weights=None, xrange=None):
                dup_last = lambda a: np.append(a, a[-1])
                
                hist, bins = np.histogram(data, bins, weights=weights, range=xrange)
                hist_raw, _ = np.histogram(data, bins)
                hist_err = np.sqrt(hist_raw) # correct uncertainties for weighted events
                integral = np.sum((bins[1:] - bins[:-1])*hist)
                scale = 1/integral
                
                ax.step(bins, dup_last(hist)*scale, label=label, linewidth=1.0, where="post", color=color)
                ax.fill_between(bins, dup_last(hist+hist_err)*scale, dup_last(hist-hist_err)*scale,
                            facecolor=color, step="post", alpha=.3)
                return bins
                
            fig4, axs4 = plt.subplots(1,2,figsize=(16,6))

            xrange = (75, 110)
            bins = plot_hist(axs4[0], test_GEN[:,9], label="LO", color="b", bins=40, weights=None, xrange=xrange)
            plot_hist(axs4[0], test_REAL[:,9], label="NLO", color="g", bins=bins, weights=None)
            plot_hist(axs4[0], test_GEN[:,9], label="Rew. LO", color="r", bins=bins, weights=weights_GENtoDATA)
            axs4[0].legend()
            axs4[0].set_xlim(xrange)
            axs4[0].set_xlabel(r"$M_ll$ of leading top")

            
            pdf.savefig(fig4)  # Save the loss curves to the PDF
            plt.close(fig4)
            """
            


            
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
            
            channels = get(self.params, "channels", None)
            if channels is None:
                channels = np.array([i for i in range(self.n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            
            label = ["REWEIGHTGEN GEN", "GEN", "DATA"]
            
           
            # Plot histograms for the first 20 dimensions
            for i, channel in enumerate(channels):
                obs_train = test_REAL[:,i]
                obs_test = test_GEN[:,i]
                obs_generated = test_GEN[:,i]
                obs_name = self.obs_names[channel]
                obs_range = self.obs_ranges[channel]
                
                
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
                            predict_weights=weights_GENtoDATA,
                            lab = label)
            
            
            
            obs_name = "M_{\ell \ell}"
            obs_range = [75,110]
            bin_num = 40
            data_train = test_REAL[:,9]
            data_test = test_GEN[:,9]
            data_generated = test_GEN[:,9]
            print(f"data generated shappe{data_generated.shape}")

            plot_obs(pp=pdf,
                        obs_train=data_train,
                        obs_test=data_test,
                        obs_predict=data_generated,
                        name=obs_name,
                        n_epochs=0,
                        range=obs_range,
                        n_jets=1,
                        weight_samples=1,
                        predict_weights=weights_GENtoDATA,
                        lab = label)  
            

            