import argparse
import os
import sys

# Add the models directory to Python path
sys.path.append(os.path.abspath("SuperGluePretrainedNetwork"))

# Import necessary models
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from GroundTruthGenerator import GroundTruthGenerator

def main():
    parser = argparse.ArgumentParser(description='Run SuperGlue-based ground truth generation.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root folder containing "rgb" and "thermal"')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder for ground truth files')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get pair IDs from the RGB folder
    rgb_dir = os.path.join(args.dataset_dir, "rgb")
    pair_ids = [os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
    
    # Instantiate the GroundTruthGenerator
    generator = GroundTruthGenerator(args.dataset_dir, args.output_dir, pair_ids)

    generator.run()

if __name__ == "__main__":
    main()