
class Config:
    def __init__(self, **kwargs):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-3
        self.epochs = 30
        self.batch_size = 16
        self.resample_rate = 16000
        
        # Dataset paths
        self.root_path = "./"
        self.vctk_path = "./vctk/"
        self.micemouse_path = "../gen/csv"
        
        # Model params
        self.n_mels = 80
        self.embed_dim = 384
        self.n_head = 6
        self.n_layer = 4
        self.kernel_size = 15
        
        # Override defaults with kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
