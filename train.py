import argparse
import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

def main():
    parser = argparse.ArgumentParser(description="Universal training script")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["gan", "diffusion", "vae", "cnn"],
                       help="Model to train")
    parser.add_argument("--variant", type=str, default="base",
                       choices=["base", "gan", "diffusion", "vae"],
                       help="CNN variant (only for CNN model)")
    
    args = parser.parse_args()
    
    if args.model == "gan":
        from gan.train import train_gan_model
        train_gan_model()
    elif args.model == "diffusion":
        from diffusion.train import train_diffusion_model
        train_diffusion_model()
    elif args.model == "vae":
        from vae.train import vae_train_model 
        vae_train_model()
    elif args.model == "cnn":
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT / 'src')
        subprocess.run([sys.executable, str(PROJECT_ROOT / 'src/cnn/train.py'), '--model', args.variant], env=env)

if __name__ == "__main__":
    main()