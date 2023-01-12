
if get(self.prior_params, "need_prior_model", False):
    self.prior_prior_path = get(self.prior_params, "prior_path", None)
    self.prior_prior_params = load_params(os.path.join(self.prior_prior_path, "paramfile.yaml"))
    self.prior_prior_model = self.load_prior_model(self.prior_prior_params, self.prior_prior_path)


if self.conditional:
    self.prior_sample = self.prior_model.sample_n(n_samples + 2 * self.batch_size, con_depth=self.con_depth)
if self.prior_model is not None:
    if self.prior_prior_model is not None:
        self.prior_prior_samples = self.prior_prior_model.sample_n(n_samples + 2 * self.batch_size,
                                                                   con_depth=self.con_depth)
    else:
        self.prior_prior_samples = None

    self.prior_samples = self.prior_model.sample_n(n_samples + self.batch_size, con_depth=self.con_depth,
                                                   prior_samples=self.prior_prior_samples)
else:
    self.prior_samples = None

    if self.n_jets == 3:
        obs_name = "\Delta R_{j_1 j_3}"
        obs_train = delta_r(plot_train[j], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[j], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[j], idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                idx_eta2=18)
        plot_obs(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 n_epochs=n_epochs,
                 n_jets=j + self.n_jets,
                 conditional=self.conditional,
                 range=[0, 8])
        obs_name = "\Delta R_{j_2 j_3}"
        obs_train = delta_r(plot_train[j], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[j], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[j], idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                idx_eta2=18)
        plot_obs(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 n_epochs=n_epochs,
                 n_jets=j + self.n_jets,
                 conditional=self.conditional,
                 range=[0, 8])

        if self.n_jets == 3:
            file_name = f"plots/run{self.runs}/deta_dphi_jets_{j + self.n_jets}_13.pdf"
            plot_deta_dphi(file_name=file_name,
                           data_train=plot_train[j],
                           data_test=plot_test[j],
                           data_generated=plot_samples[j],
                           idx_phi1=9,
                           idx_phi2=17,
                           idx_eta1=10,
                           idx_eta2=18,
                           n_jets=j + self.n_jets,
                           conditional=self.conditional,
                           n_epochs=n_epochs)

            file_name = f"plots/run{self.runs}/deta_dphi_jets_{j + self.n_jets}_23.pdf"
            plot_deta_dphi(file_name=file_name,
                           data_train=plot_train[j],
                           data_test=plot_test[j],
                           data_generated=plot_samples[j],
                           idx_phi1=13,
                           idx_phi2=14,
                           idx_eta1=10,
                           idx_eta2=18,
                           n_jets=j + self.n_jets,
                           conditional=self.conditional,
                           n_epochs=n_epochs)