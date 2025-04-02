# fit-repository

A Flask-based web application for virtual clothing try-on using 3D body models.

## Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA-capable GPU (CUDA 11.3 recommended)
- OpenGL support

## Installation

1. Clone the repository and required dependencies:
```bash
# Clone main repository
git clone https://github.com/yourusername/fit-repository.git
cd fit-repository

# Clone SMPLX repository (required for body model)
git clone https://github.com/vchoutas/smplx.git resources/smplx
cd resources/smplx
python setup.py install
cd ../..

# Clone SHAPY repository
git clone https://github.com/muelea/shapy.git resources/shapy
cd resources/shapy
python setup.py install
cd ../..

# Clone DrapeNet repository
git clone https://github.com/liren2515/DrapeNet.git resources/drapenet
cd resources/drapenet
python setup.py install
cd ../..
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate fitrepository
```

3. Download required resources:
Due to size limitations, model checkpoints and resources are not included in the repository. Please download them from the following locations:

- DRAPENet resources: [Download Link]
  - Extract to `resources/drapenet/`
  - Required files:
    - `extra-data/pose-sample.pt`
    - `smpl_pytorch/`
    - `checkpoints/`
      - `top_udf.pt`
      - `bottom_udf.pt`
      - `bottom_codes.pt`
- SHAPY resources: [Download Link]
  - Extract to `resources/shapy/`
  - Required files:
    - `data/trained_models/`
    - `data/body_models/`
- SMPL Model files: [Download Link]
  - Extract to `resources/smplx/models/`
  - Required files:
    - `smpl/`
    - `smplx/`
- Checkpoints: [Download Link]
  - Extract to `resources/checkpoints/`
  - Required files:
    - `pointcount_prediction_model.pth`
    - `best_model.pth`

## Configuration

Create a `.env` file in the project root with the following variables (adjust as needed):
```
CUDA_DEVICE=0
OUTPUT_DIR=static/output
EXTRA_DIR=resources/drapenet/extra-data
CHECKPOINTS_DIR=resources/checkpoints
SMPL_MODEL_DIR=resources/smplx/models
```

## Running the Application

1. Activate the conda environment:
```bash
conda activate fitrepository
```

2. Start the Flask server:
```bash
python project/app.py
```

3. Access the application at `http://localhost:5001`

## Project Structure

```
fit-repository/
├── project/
│   ├── app.py              # Main Flask application
│   ├── models/             # Model implementations
│   └── templates/          # HTML templates
├── resources/
│   ├── checkpoints/        # Model checkpoints
│   ├── drapenet/          # DrapeNet repository and resources
│   │   ├── extra-data/    # DrapeNet extra data
│   │   ├── smpl_pytorch/  # SMPL PyTorch implementation
│   │   └── checkpoints/   # DrapeNet model checkpoints
│   ├── shapy/             # SHAPY repository and resources
│   │   ├── data/         # SHAPY trained models and body models
│   │   └── attributes/   # SHAPY attribute models
│   └── smplx/             # SMPLX repository and models
│       └── models/        # SMPL and SMPLX model files
├── static/                 # Static files and output directory
│   └── output/
├── environment.yml         # Conda environment specification
└── README.md
```

## Notes

- Large resource files are not included in the repository. Please download them separately using the provided links.
- Make sure to have sufficient disk space for the resource files (approximately 10GB).
- The application requires a CUDA-capable GPU for optimal performance.
- If you encounter OpenGL-related issues, make sure you have the appropriate graphics drivers installed.

## License

This project incorporates code from multiple sources with different licenses:
- SHAPY: [SHAPY License](https://github.com/muelea/shapy/blob/master/LICENSE)
- DrapeNet: [DrapeNet License](https://github.com/liren2515/DrapeNet/blob/master/LICENSE)
- SMPL-X: [SMPL-X License](https://github.com/vchoutas/smplx/blob/master/LICENSE)

Please make sure to comply with all respective licenses when using this code.
