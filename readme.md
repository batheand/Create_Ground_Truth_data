# Ground Truth Generator

This project generates ground truth files for RGB-thermal image pairs using SuperGlue and SuperPoint. 

## Setup Instructions

### 1. Clone the SuperGluePretrainedNetwork Repository

SuperGlue and SuperPoint models are required for feature matching. Clone the repository and place it in your project directory:

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```

### 2. Install Dependencies

Ensure you have Python installed, then install the required packages:

```bash
pip install numpy opencv-python torch torchvision matplotlib
```

### 3. Verify SuperGlue Model Directory

Ensure that `SuperGluePretrainedNetwork` is placed in your project directory. The following files should exist:

```
SuperGluePretrainedNetwork/
│── models/
│   ├── superpoint.py
│   ├── superglue.py
```

## Running the Script

Run `main.py` with the dataset directory and output directory as arguments:

```bash
python main.py --dataset_dir <path-to-dataset> --output_dir <path-to-output>
```

Example:

```bash
python main.py --dataset_dir ./data --output_dir ./output
```

## Dataset Structure

The dataset should be organized as follows:

```
<data-root>/
│── rgb/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│── thermal/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
```

- `rgb/` contains visible light images.
- `thermal/` contains corresponding thermal images.
- Filenames in both folders should match (e.g., `image1.jpg` in both).

## Output Format

Generated files will be stored in the output directory:

```
<output-dir>/
│── image1.txt
│── image2.txt
│── ...
```

Each `.txt` file contains processed SuperGlue data.
