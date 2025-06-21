import argparse
import sys
import os
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
        from gan.generate import generate
        generate()
    elif args.model == "diffusion":
        from diffusion.generate import generate
        generate()
    # elif args.model == "vae":
        # from vae.generate import generate 
        # generate()

if __name__ == "__main__":
    main()