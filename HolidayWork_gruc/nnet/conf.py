fs = 16000
chunk_len = 4  # (s)
chunk_size = fs * chunk_len 
num_spks = 1

# network configure
nnet_conf = {
}

# data configure:
train_dir = "data/tr/"
dev_dir = "data/cv/"

train_data = {
    "mix_scp":  train_dir + "mix.scp",
    "ref_scp": [train_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

dev_data = {
    "mix_scp":  dev_dir + "mix.scp",
    "ref_scp": [dev_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 50  # batch number
}