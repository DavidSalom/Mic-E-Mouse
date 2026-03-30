
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
import einops
import os
import whisper

# Import our new modules
from core.config import Config
from core.nn.reconstruction import ReconstructionModel
import core.nn.VCTK as VCTK
from core.nn.utils import buildTransforms, normalize, denormalize, apply_wiener, computeSpec, computeUnboundedSpec, fromSpec
# We need to import preprocess to ensure everything is initialized if needed, though mostly used via VCTK
from core.signal.preprocess import processFromFile

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Mic-E-Mouse Reconstruction Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dry_run", action="store_true", help="Run a single batch for testing")
    parser.add_argument("--noise_path", type=str, default="../gen/noise/noise_2023-11-03_21-00-40.csv", help="Path to noise file")
    parser.add_argument("--inference", action="store_true", help="Run in inference mode")
    parser.add_argument("--load_model", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to save outputs")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(**vars(args))
    
    # Setup logging
    logging_level = logging.ERROR if args.silent else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)s %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create cache dir
    os.makedirs("cache", exist_ok=True)

    # --- Data Setup ---
    Fs = cfg.resample_rate
    
    groundTruthConfig = {
        "root": cfg.vctk_path,
        "download": True, # Ensure it downloads if missing
    }
    sensorConfig = {
        "root": cfg.micemouse_path,
        "Fs": Fs,
        "device": device,
        "resampleMethod": 'sinc'
    }
    
    try:
        logging.info("Loading Dataset...")
        paired_ds = VCTK.PairedAudioDataset(groundTruthConfig, sensorConfig)
        # Note: removeOutliers might take a while on first run as it calculates lengths
        filtered_paired_ds = VCTK.removeOutliers(paired_ds, Fs=Fs)
        train_loader, test_loader, dev_loader = VCTK.getAudioLoaders(
            filtered_paired_ds, 
            device=device, 
            batch_size=cfg.batch_size,
            Fs=Fs
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.error("Please check dataset paths in core/config.py or arguments.")
        return

    if args.dry_run:
        logging.info("Dry run mode: limiting to 1 batch")
        
    # --- Transforms and Wiener Filter Setup ---
    tforms = buildTransforms(device=device, Fs=Fs, n_fft=800, win_length=800, hop_length=160, n_mels=80)
    specFn, inverseSpecFn, melFilterMouse, melFilter, inverseMelFn = tforms


    # --- Wiener Filter and Stats Calculation ---
    if os.path.exists("cache/H.pt") and os.path.exists("cache/stats.pt"):
        logging.info("Loading cached Wiener filter and stats...")
        H = torch.load("cache/H.pt", map_location=device)
        stats = torch.load("cache/stats.pt")
        mnmn, mxmx = stats['mnmn'], stats['mxmx']
    else:
        logging.info("Computing Wiener Filter and Stats from dev set (this may take a moment)...")
        # 1. Compute Noise FFT
        # We need to load noise file from args.noise_path
        # Assuming we can resolve it. If not, fallback or error.
        try:
            # Need to fix path if relative from notebook location
            # (Truncated logic from previous step, keeping successful path resolution code)
            noise_path = args.noise_path
            if not os.path.exists(noise_path):
                noise_path = os.path.abspath(noise_path)
            if not os.path.exists(noise_path):
                 noise_path = os.path.abspath("../gen/noise/noise_2023-11-03_21-00-40.csv")
            
            if not os.path.exists(noise_path):
                logging.warning(f"Noise file not found at {noise_path}. Wiener filter will be suboptimal (ones).")
                H = torch.ones(5 * Fs // 2 + 1).to(device)
            else:
                from math import ceil
                _, Xnoise, Ynoise = processFromFile(noise_path, method='sinc')
                noise_sig = (Xnoise + Ynoise) / 2
                target_len = 5 * Fs
                P = ceil((noise_sig.shape[0] / target_len)) * target_len - noise_sig.shape[0]
                noise_sig = F.pad(noise_sig, (0, P), "constant", 0)
                noise_sig = noise_sig.reshape(-1, target_len).to(device)
                
                noiseFFT = torch.fft.rfft(noise_sig, dim=1)
                noiseFFT = torch.abs(noiseFFT)
                noiseFFT = noiseFFT.mean(dim=0)
                
                Data_PSD = []
                for Xorig, Yorig in tqdm(dev_loader, desc="Computing PSD"):
                    Y_fft = torch.fft.rfft(Yorig.to(device), dim=-1).abs().mean(dim=0)
                    Data_PSD.append(Y_fft)
                    if args.dry_run: break
                
                avgData_PSD = torch.stack(Data_PSD).mean(dim=0)
                
                if noiseFFT.shape[-1] != avgData_PSD.shape[-1]:
                    logging.warning("Shape mismatch in Wiener Filter calc. Resizing noise PSD.")
                    noiseFFT = F.interpolate(noiseFFT.unsqueeze(0).unsqueeze(0), size=avgData_PSD.shape[-1], mode='linear').squeeze()
                
                H = avgData_PSD / (avgData_PSD + noiseFFT + 1e-8)
                torch.save(H, "cache/H.pt")
                
        except Exception as e:
            logging.error(f"Error computing Wiener Filter: {e}")
            H = torch.ones(5 * Fs // 2 + 1).to(device)

        # 2. Compute Normalization Stats
        logging.info("Computing Min/Max stats...")
        mnmn = torch.inf
        mxmx = -torch.inf
        
        for Mwav, Wwav in tqdm(dev_loader, desc="Computing Stats"):
            # Mwav: [B, 2, T], Wwav: [B, T]
            Mwav, Wwav = Mwav.to(device), Wwav.to(device)
            MwavFilt = apply_wiener(Mwav, H)
            
            Mfilt = computeUnboundedSpec(MwavFilt, specFn, melFilterMouse, melFilter)
            M = computeUnboundedSpec(Mwav, specFn, melFilterMouse, melFilter)
            W = computeUnboundedSpec(Wwav, specFn, melFilterMouse, melFilter)
            
            mnmn = min(mnmn, M.min().item(), W.min().item(), Mfilt.min().item())
            mxmx = max(mxmx, M.max().item(), W.max().item(), Mfilt.max().item())
            if args.dry_run: break
            
        torch.save({'mnmn': mnmn, 'mxmx': mxmx}, "cache/stats.pt")
        
    logging.info(f"Stats: Min={mnmn}, Max={mxmx}")

    def computeSpec(X):
        # We need to pass the transforms available in this scope
        X = computeUnboundedSpec(X, specFn, melFilterMouse, melFilter)
        X = (X - mnmn) / (mxmx - mnmn)               # Map to bounds
        X = torch.clamp(X, 0, 1)                    # Clamp to [0, 1]
        X = 2 * X - 1                               # Map to [-1, 1]
        return X

    # --- Model Setup ---
    model = ReconstructionModel(
        n_mels=cfg.n_mels,
        embed_dim=cfg.embed_dim,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        kernel_size=cfg.kernel_size
    ).to(device)
    
    if args.inference:
        if not args.load_model:
            logging.error("Inference mode requires --load_model argument.")
            # return
            # Allow fallback to dummy model for dry run test
            if not args.dry_run: return
            logging.warning("Continuing with untrained model for dry run.")
        else:
            if os.path.exists(args.load_model):
                logging.info(f"Loading model from {args.load_model}")
                try:
                    state_dict = torch.load(args.load_model, map_location=device)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    logging.error(f"Failed to load model weights: {e}")
                    return
            else:
                logging.error(f"Model file not found: {args.load_model}")
                return
        
        model.eval()
        logging.info("Starting Inference...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Helper to invert spec
        from core.nn.utils import inverseBounds, fromMelDB
        
        def invert_pipeline(spec):
            # spec: [B, C*n_mels, T] -> needs undoing what computeSpec did.
            # 1. Inverse Bounds
            spec = inverseBounds(spec, mnmn, mxmx)
            # 2. fromMelDB (Log -> Mel -> Stft -> GriffinLim)
            # fromMelDB expects [B, n_mels, T] usually.
            # If our model output is [B, 80, T], we treat it as 1 channel.
            return fromMelDB(spec, inverseMelFn, inverseSpecFn)

        count = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                mic, wav = batch
                mic, wav = mic.to(device), wav.to(device)
                
                # Preprocess
                mic_filt = apply_wiener(mic, H)
                X = computeSpec(mic_filt)
                
                # Inference
                recon_spec = model(X)
                
                # Invert to Audio
                recon_audio = invert_pipeline(recon_spec)
                
                # Save results
                for j in range(len(recon_audio)):
                    if count >= args.num_samples: break
                    
                    # Original GT
                    torchaudio.save(f"{args.output_dir}/sample_{count}_gt.wav", wav[j].unsqueeze(0).cpu(), Fs)
                    # Reconstructed
                    torchaudio.save(f"{args.output_dir}/sample_{count}_recon.wav", recon_audio[j].unsqueeze(0).cpu(), Fs)
                    
                    count += 1
                
                if count >= args.num_samples: break
                
        logging.info(f"Inference Done. Saved {count} samples to {args.output_dir}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.707)
    
    # --- Loss Setup ---
    l2_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss()
    
    # Perceptual Loss
    try:
        perceptualEmbedding = whisper.load_model("tiny").encoder.to(device)
        perceptualEmbedding.eval()
        for p in perceptualEmbedding.parameters():
            p.requires_grad = False
        logging.info("Whisper model loaded for Perceptual Loss.")
        
        def perceptual_loss(x, y):
            # Whisper expects 3000 frames? Or specific input?
            # Model output is [B, n_mels=80, T=501]
            # Whisper expects [B, 80, 3000]. We pad.
            pad_len = 3000 - x.shape[-1]
            if pad_len > 0:
                x = F.pad(x, (0, pad_len))
                y = F.pad(y, (0, pad_len))
            return l2_loss(perceptualEmbedding(x), perceptualEmbedding(y))
            
    except Exception as e:
        logging.warning(f"Could not load Whisper model ({e}). Fallback to simple loss.")
        perceptual_loss = lambda x, y: 0.0

    loss_fn = lambda x, y: (perceptual_loss(x, y) + huber_loss(x, y))

    # --- Training Loop ---
    logging.info("Starting Training...")
    model.train()
    
    for epoch in range(cfg.epochs):
        loop = tqdm(train_loader, disable=args.silent)
        running_loss = 0.0
        
        for i, batch in enumerate(loop):
            optimizer.zero_grad()
            
            # batch is [mic, wav] (collated)
            # mic: [B, 2, T], wav: [B, T]
            mic, wav = batch
            mic, wav = mic.to(device), wav.to(device)

            # Preprocessing
            mic = apply_wiener(mic, H)
            
            # Determine target and input
            # Input: Processed Mic Spectrogram
            # Target: Ground Truth Wav Spectrogram
            
            with torch.no_grad():
                X = computeSpec(mic)
                Y = computeSpec(wav)
            
            # Model forward
            # Input to model expects [B, 2*n_mels, T_spec] ?
            # computeSpec returns [B, C*n_mels, T_spec] basically.
            # ReconstructionModel expects [B, 2*n_mels, T]
            # Let's check shapes.
            # computeUnboundedSpec for 'ms' (mouse, 2 channel):
            #   X = einops.rearrange(X, "b c m t -> b (m c) t", c=2)
            # So shape is [B, 2*80, T_spec].
            # ReconstructionModel definition:
            #   nn.Conv1d(n_mels * 2, embed_dim, ...)
            # So shapes align perfectly. 
            # Note: For ground truth Y, it is 1 channel, so [B, 1*80, T].
            # ReconstructionModel output:
            #   nn.Conv1d(n_mels * 4, n_mels, 1, padding=0) -> Output channels = n_mels (80).
            # So Model(X) -> [B, 80, T].
            # Y -> [B, 80, T].
            # Loss compares them.
            
            recon = model(X) 
            loss = loss_fn(recon, Y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{cfg.epochs} Loss: {loss.item():.4f}")
            
            if args.dry_run:
                break
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader) 
        logging.info(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"models/model_E{epoch+1}_L{avg_loss:.3f}.pt")
        
        if args.dry_run:
            logging.info("Dry run complete.")
            break
            
    logging.info("Training Done.")

if __name__ == "__main__":
    main()