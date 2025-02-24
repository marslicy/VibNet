config_list = [
    {
        "expriment_name": "beef",
        "data": {
            "data_path": "dataset/Beef",
            "size": (657 // 2, 671 // 2),
        },
        "train": {
            "device": "cuda:0",
            "val_every_n": 300,
            "print_every_n": 25,
            "lr": 1e-4,
            "w_shaft": 0.95,
            "w_tip": 0.05,
            "batch_size_train": 6,
            "batch_size_val": 8,
            "epoch": 10,
            "early_stop": 10,
        },
        "model": {
            "seq_length": 30,
            "num_angle": 180,
            "num_rho": 100,
            # Items after this line could be removed if you just want to use the default value
            "win": 10,
            "stride": 3,
            "FocalLoss": True,
            "enc_init": True,
            "fic_init": True,
        },
    },
]
