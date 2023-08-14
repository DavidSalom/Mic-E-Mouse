import argparse
import logging
import diffusers.schedulers

class Config:
    def __init__(self, args):
        logging.info("Parsing arguments...")
        
        parser = argparse.ArgumentParser(description='Train a CrossAttentionWavUNet model on a given dataset.')
        
        # Training Parameters
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for.')
        parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
        parser.add_argument('--device', type=str, default="cuda", help='Device to train on.')
        parser.add_argument('--validation', type=bool, default=False, help='Whether to use validation or train on the whole dataset.')
        
        # Audio Preprocessing Parameters
        parser.add_argument('--resample_rate', type=int, default=8000, help='Resample rate.')
        parser.add_argument('--clip_min', type=float, default=-10.4615, help='Minimum value to clip to.')
        parser.add_argument('--clip_max', type=float, default=11.3003, help='Maximum value to clip to.')
        
        # Dataset Parameters
        parser.add_argument('--train_split', type=float, default=0.9, help='Fraction of the dataset to use for training.')
        parser.add_argument('--val_split', type=float, default=0.05, help='Fraction of the dataset to use for validation.')
        parser.add_argument('--test_split', type=float, default=0.05, help='Fraction of the dataset to use for testing.')
        
        # Meta Parameters
        parser.add_argument('--debug', type=bool, default=False, help='Whether to run in debug mode.')
        parser.add_argument('--silent', type=bool, default=False, help='Whether to run in silent mode.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--VCTK_root_path', type=str, default="/mnt/C/VCTK_PREPROCESSED_8K/", help='Path to the VCTK dataset parent folder, which should contain wav48 and txt folders.')
        parser.add_argument('--save_path', type=str, default="./workspace", help='Path to save the models to.')
        parser.add_argument('--load_path', type=str, default=None, help='Path to load the model from.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader.')
        parser.add_argument('--wandb', type=bool, default=False, help='Whether to use wandb.')
        
        # Diffusion Parameters
        parser.add_argument('--N_diffusion', type=int, default=1000, help='Number of diffusion steps.')
        parser.add_argument('--beta_min', type=float, default=1e-4, help='Minimum value of beta for theDiffusion process.')
        parser.add_argument('--beta_max', type=float, default=0.005, help='Maximum value of beta for the Diffusion process.')
        parser.add_argument('--scheduler', type=str, default="DDPMScheduler", help="Which Diffusers Library Scheduler to use (DDPMScheduler, KDPM2AncestralDiscreteScheduler, ...)")
        
        args_parsed = parser.parse_args(args)
        for arg in vars(args_parsed):
            self.__dict__[arg] = getattr(args_parsed, arg)
        
        assert self.scheduler in diffusers.schedulers.__dict__, "scheduler must be a diffusers library scheduler"
        
        assert self.train_split + self.val_split + self.test_split == 1.0, "train_split + val_split + test_split must equal 1.0"
        logging.info("Done parsing arguments. Checks passed.")