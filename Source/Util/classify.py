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
    def __init__(self, n_layers=8, dim_in=10, n_hidden=256, dropout=0.1):
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
 
        #self.data_real = np.load("/remote/gpu03/hoelzl/data09/ttbarj_LO.npy")
        #self.data_gen =  np.load("/remote/gpu03/hoelzl/data09/ttbarj_NLO.npy")

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
        n1 = 2_000_000
        #n_val = int(0.5 * (total_data_points - n1))  # 50% of the remaining 40% for validation
        n2 = 2_500_000

        #n1 = 50_000 
        #n2 = 60_000 

        batchsize = self.BATCHSIZE
        self.data_trn, self.data_val, self.data_tst = self.data[:n1,:], self.data[n1:n2,:], self.data[n2:,:]
        print(f"TRAIN_DATA SHAPE: {self.data_trn.shape}")
        print(f"VAL_DATA SHAPE: {self.data_val.shape}")
        print(f"TEST_DATA SHAPE: {self.data_tst.shape}")
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
        print(f"train_model: Training Class Model on {self.label}...")

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

        patience = 10
        es_epochs = 0
        min_val_loss = 1e20

        for epoch in range(class_epochs):
            train_epoch(self.loader_trn, self.losses)

            val_loss = val_epoch(self.loader_val)
            self.val_loss.append(val_loss)
            print(f"{epoch}/{class_epochs} val_loss:{val_loss:0.5f}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                es_epochs = 0
            else:
                es_epochs += 1
                if es_epochs == patience:
                    print(f"Early stopping in epoch {epoch} after no improvement in {es_epochs} epochs")
                    break

        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/CLASSmodel.pt")

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
            ax1.hist(pred_sig[truth == 0], range=(0, 1), bins=100, alpha=0.4, label=self.label[0], color='blue') #density=True
            ax1.hist(pred_sig[truth == 1], range=(0, 1), bins=100, alpha=0.4, label=self.label[1], color='red')
            #ax1.set_yscale('log')
            ax1.legend()
            ax1.set_xlim([0.0, 1.0])
            ax1.set_title(f'{self.label[0]} Histogram of predicted events')
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
            ax2.set_title(f'{self.label[0]},{self.label[1]} Receiver Operating Characteristic (ROC) Curve: AUC = {auc_score:.3f}')
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
            ax3.set_title(f'{self.label[0]},{self.label[1]} Loss Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            pdf.savefig(fig3)  # Save the loss curves to the PDF
            plt.close(fig3)

            #cal WEIGHTS
            weights = np.exp(pred)
            print(f"all weights shape {weights.shape}")

            #PLOTTING DATA; GEN; AND REWEIGHTED DATA
            test_LO = self.data_tst[self.labels_tst[:,0]==0]
            test_NLO = self.data_tst[self.labels_tst[:,0]==1]

            mass_top1_LO = test_LO[:,9]
            mass_top1_NLO = test_NLO[:,9]

            weights_LOtoNLO = weights[self.labels_tst[:,0]==0]

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

            for i,channel in enumerate(channels):
                obs_name = self.obs_names[channel]
                obs_range = self.obs_ranges[channel]
                unit = None

                fig, ax = plt.subplots(figsize=(10, 6))
                xrange1 = obs_range
                bins1 = plot_hist(ax, test_LO[:, i], label=self.label[0], color="black", bins=40, weights=None, xrange=xrange1)
                plot_hist(ax, test_NLO[:, i], label=self.label[1], color="#A52A2A", bins=bins1, weights=None)
                plot_hist(ax, test_LO[:, i], label="Reweighted", color="#0343DE", bins=bins1, weights=weights_LOtoNLO)
                ax.legend()
                ax.set_xlim(xrange1)
                ax.set_xlabel(r"${%s}$ %s" % (obs_name, ("" if unit is None else f"[{unit}]")), fontsize = 16)
                pdf.savefig(fig)  # Save the first histogram to the PDF
                plt.close(fig)

            # Second figure
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            xrange2 = (75, 110)
            bins2 = plot_hist(ax2, mass_top1_LO, label="Truth", color="black", bins=40, weights=None, xrange=xrange2)
            plot_hist(ax2, mass_top1_NLO, label="Gen", color="#A52A2A", bins=bins2, weights=None)
            plot_hist(ax2, mass_top1_LO, label="Reweighted", color="#0343DE", bins=bins2, weights=weights_LOtoNLO)
            ax2.legend()
            ax2.set_xlim(xrange2)
            ax2.set_xlabel(r"$M_{ll}$")
            pdf.savefig(fig2)  # Save the second histogram to the PDF
            plt.close(fig2)