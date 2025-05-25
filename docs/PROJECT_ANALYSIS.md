# Autonomous Driving Project Analysis

This document provides a comprehensive analysis of the autonomous driving project, which is a PyTorch implementation inspired by the paper "End to End Learning for Self-Driving Cars" by NVIDIA. The project aims to train a model to predict steering angles from camera images.

## 1. Project Goal

The primary goal of this project is to develop and train a deep learning model (Convolutional Neural Network) that can predict steering angles for a self-driving car based on camera images. This enables end-to-end learning, where the model learns to map raw visual input directly to driving control commands, primarily for the CARLA simulator.

## 2. Core Components and Files

Here's a breakdown of the key Python files and their roles:

*   **`config.py`**:
    *   **Purpose**: Centralized configuration for the project.
    *   **Key Functionalities**: Defines default parameters for:
        *   Dataset settings (image dimensions, dataset types).
        *   Training parameters (batch size, number of epochs, learning rate, optimizer settings).
        *   Model saving paths (directory for trained models).
        *   Device selection (CUDA or CPU).
        *   Normalization statistics (mean and standard deviation, though these might also be handled elsewhere).
    *   **Interaction**: Read by various scripts like `train.py`, `validate.py`, `model.py`, and `dataset_loader.py` to ensure consistent settings.

*   **`dataset_loader.py`**:
    *   **Purpose**: Handles loading, preprocessing, and augmentation of datasets.
    *   **Key Functionalities**:
        *   `UdacitySimulatorDataset`: PyTorch `Dataset` class for Udacity simulator data. Handles loading images (center, left, right) and steering angles from a CSV log (`driving_log.csv`). Includes logic to adjust steering angles for left/right cameras and applies data augmentation.
        *   `CarlaSimulatorDataset`: PyTorch `Dataset` class for CARLA simulator data. Loads images and steering angles from a CSV log (e.g., `steering_data.csv`). Applies data augmentation.
        *   Augmentation functions: `add_random_shadow_bgr`, `add_random_brightness_bgr` (applied before YUV conversion). Images are also flipped horizontally with corresponding steering angle inversion.
        *   Image preprocessing: Converts images to YUV color space, resizes them to (66, 200), and normalizes pixel values to `[-1.0, 1.0]` before converting to PyTorch tensors.
        *   `get_inference_dataset()`: Returns a single `Dataset` object based on the specified `dataset_type`.
        *   `get_datasets()`: Concatenates multiple specified datasets into a single `ConcatDataset`.
        *   `get_data_subsets_loaders()`: Creates and returns `DataLoader` instances for training and validation sets by splitting the concatenated dataset.
        *   `get_full_dataset_loader()`: Returns a `DataLoader` for a single, full dataset (used by `calculate_dataset_std_mean.py`).
    *   **Interaction**: Used by `train.py`, `validate.py`, `cross_validation.py`, `calculate_dataset_std_mean.py`, and `inference_run_dataset.py` to access and prepare data.

*   **`model.py`**:
    *   **Purpose**: Defines the neural network architecture(s).
    *   **Key Functionalities**:
        *   `NvidiaModel`: Implements the CNN architecture based on the NVIDIA paper. It consists of convolutional layers followed by fully connected layers. Includes an initial normalization layer and a cropping layer.
        *   The model architecture: 5 convolutional layers with ReLU activations (some with strides for downsampling), followed by flattening and 3 fully connected layers. Dropout is applied after the first fully connected layer.
    *   **Interaction**: Instantiated and used in `train.py`, `validate.py`, `cross_validation.py`, and inference scripts.

*   **`train.py`**:
    *   **Purpose**: Main script for training the neural network model.
    *   **Key Functionalities**:
        *   Parses command-line arguments for training parameters (e.g., dataset types, epochs, batch size, learning rate), allowing overrides of `config.py` settings.
        *   Initializes TensorBoard for logging training metrics (loss, learning rate).
        *   Loads training and validation data using `dataset_loader.get_data_subsets_loaders()`.
        *   Instantiates the `NvidiaModel`.
        *   Sets up the Adam optimizer and a learning rate scheduler (ReduceLROnPlateau).
        *   Uses Mean Squared Error (MSE) as the loss function.
        *   Implements an early stopping mechanism (`utils.EarlyStopping`) to prevent overfitting by monitoring validation loss.
        *   Contains the main training loop: iterates over epochs, calls `utils.train_epoch()` for training and `utils.validate_epoch()` for validation.
        *   Saves the best performing model (based on validation loss) to the path specified in `config.py`.
    *   **Interaction**: Orchestrates the training process by using `dataset_loader.py`, `model.py`, and `utils.py`.

*   **`validate.py`**:
    *   **Purpose**: Script for evaluating a trained model on a validation dataset.
    *   **Key Functionalities**:
        *   Loads a pre-trained model from a specified path (`./save/model.pt` by default).
        *   Loads validation data using `dataset_loader.get_data_subsets_loaders()`, configured to use a specific dataset type (e.g., 'carla_001').
        *   Calculates and prints the Mean Squared Error (MSE) loss on the validation set.
        *   Saves batch loss means to a CSV file (`loss_acc_results_validation.csv`).
    *   **Interaction**: Uses `dataset_loader.py`, `model.py`, and `config.py`.

*   **`utils.py`**:
    *   **Purpose**: Contains various helper functions used across the project.
    *   **Key Functionalities**:
        *   `save_model()`: Saves the model's state dictionary.
        *   `train_epoch()`: Logic for training the model for one epoch (forward pass, loss calculation, backpropagation, optimizer step, metric logging).
        *   `validate_epoch()`: Logic for validating the model for one epoch (forward pass, loss calculation, metric logging).
        *   TensorBoard logging helpers: `add_train_val_loss_to_tensorboard`, `add_learning_rate_to_tensorboard`.
        *   `EarlyStopping`: Class to implement early stopping based on validation loss.
        *   `batch_mean_and_sd()`: (Also present in `calculate_dataset_std_mean.py`) Calculates mean and standard deviation for a dataset given a loader.
    *   **Interaction**: Provides core training/validation loops and utility functions to `train.py` and `cross_validation.py`.

*   **`calculate_dataset_std_mean.py`**:
    *   **Purpose**: Utility script to compute the mean and standard deviation of an image dataset.
    *   **Key Functionalities**:
        *   Takes a `dataset_type` as a command-line argument.
        *   Uses `dataset_loader.get_full_dataset_loader()` to load the specified dataset.
        *   Iterates through the dataset to calculate the channel-wise mean and standard deviation of the images.
        *   Prints the calculated mean and std, which can be used for input normalization if needed (though normalization is also handled in `dataset_loader.py` and `model.py`).
    *   **Interaction**: Uses `dataset_loader.py`.

*   **`cross_validation.py`**:
    *   **Purpose**: Script for performing k-fold cross-validation.
    *   **Key Functionalities**:
        *   Implements k-fold cross-validation to get a more robust evaluation of the model's performance.
        *   Splits the dataset into k folds. For each fold, it trains the model on k-1 folds and validates on the held-out fold.
        *   Uses `dataset_loader.py` to get data and `NvidiaModel` from `model.py`.
        *   Leverages `utils.train_epoch`, `utils.validate_epoch`, `utils.EarlyStopping`, and `utils.save_model`.
        *   Logs metrics for each fold to TensorBoard.
        *   Calculates and prints the average validation loss across all folds.
    *   **Interaction**: Heavily relies on `dataset_loader.py`, `model.py`, and `utils.py`.

*   **`inference_run_dataset.py`**:
    *   **Purpose**: Runs a trained model on a specified dataset to generate and optionally save steering angle predictions.
    *   **Key Functionalities**:
        *   Loads a trained model.
        *   Loads a dataset using `dataset_loader.get_inference_dataset()`.
        *   Iterates through the dataset, making predictions with the model.
        *   Saves predicted steering angles along with ground truth angles and image filenames to a CSV file.
        *   Optionally displays images with predicted and actual steering angles.
    *   **Interaction**: Uses `dataset_loader.py`, `model.py`, and `config.py`.

*   **`inference_run_video.py`**:
    *   **Purpose**: Runs a trained model on a video file to predict steering angles frame by frame and optionally save the output as a new video.
    *   **Key Functionalities**:
        *   Loads a trained model.
        *   Reads an input video file frame by frame.
        *   Preprocesses each frame (YUV conversion, resize, normalization) similar to `dataset_loader.py`.
        *   Predicts steering angle for each frame.
        *   Draws the predicted steering angle on the frame (visualized as a line on a steering wheel image).
        *   Optionally saves the processed frames as an output video.
    *   **Interaction**: Uses `model.py` and `config.py`. Requires `steering_wheel_image.jpg` for visualization.

*   **`export_model_to_onnx.py`**:
    *   **Purpose**: Converts a trained PyTorch model (saved as `.pt`) to the ONNX (Open Neural Network Exchange) format.
    *   **Key Functionalities**:
        *   Loads a PyTorch model (`NvidiaModel`).
        *   Creates a dummy input tensor with the expected input shape.
        *   Uses `torch.onnx.export()` to perform the conversion.
        *   Saves the model as an `.onnx` file.
    *   **Interaction**: Uses `model.py`. ONNX models are useful for deployment across different frameworks and hardware.

*   **`plot_loss_results.py`**:
    *   **Purpose**: Simple script to plot training and validation loss from a CSV file (presumably `loss_acc_results_train.csv` which `train.py` saves).
    *   **Key Functionalities**:
        *   Reads loss data using pandas.
        *   Uses matplotlib to create and save a plot of training vs. validation loss over epochs.
    *   **Interaction**: Standalone utility, relies on CSV output from `train.py`.

*   **`chart.py`**:
    *   **Purpose**: Appears to be a script for generating a bar chart from a CSV file containing loss values (e.g., `loss_acc_results_validation.csv`).
    *   **Key Functionalities**: Reads data using pandas and plots a bar chart using matplotlib.
    *   **Interaction**: Standalone utility, relies on CSV output.

*   **`capture_data/collect_carla_data.py`**:
    *   **Purpose**: Script to connect to a running CARLA simulator, control a vehicle (or let it run on autopilot), and record camera images and corresponding steering angles.
    *   **Key Functionalities**:
        *   Connects to CARLA server.
        *   Spawns a vehicle and a front-facing RGB camera attached to it.
        *   Collects camera images and vehicle control data (steering angle).
        *   Saves images to an `images/` subfolder and writes image filenames and steering angles to a `steering_data.csv` file within a timestamped session directory (e.g., `datasets/YYYYMMDD_HHMMSS/`).
        *   Uses threading for efficient data collection and saving.
        *   Allows configuration via command-line arguments (host, port, save directory, max frames, sync mode).
    *   **Interaction**: Interacts with the CARLA simulator. Produces datasets in the format expected by `CarlaSimulatorDataset`.

*   **`capture_data/steering_data_collection.py`**:
    *   **Purpose**: A simpler, alternative script for data collection, possibly for manual control or specific scenarios. It focuses on capturing keyboard input for steering and saving image frames.
    *   **Key Functionalities**:
        *   Connects to CARLA.
        *   Spawns a vehicle and camera.
        *   Listens for keyboard inputs (A/D for steering, W for saving).
        *   Saves images and a CSV with image names and steering angles.
    *   **Interaction**: Interacts with CARLA.

*   **`convert_to_jpg.py`**:
    *   **Purpose**: Utility script to convert images in a dataset from their original format (e.g., PNG) to JPG format. It can also resize images.
    *   **Key Functionalities**:
        *   Traverses a specified input directory.
        *   Reads images, converts them to JPG, optionally resizes them.
        *   Saves the converted images to an output directory, maintaining a similar structure.
        *   Updates an associated CSV file (like `steering_data.csv`) to point to the new JPG filenames.
    *   **Interaction**: Filesystem operations, image manipulation with OpenCV.

## 3. Project Flow (Simplified for CARLA)

1.  **Setup & Configuration**:
    *   Environment setup (Python, PyTorch, CARLA, etc.).
    *   Parameters reviewed/adjusted in `config.py`.
2.  **Data Collection (using `capture_data/collect_carla_data.py`)**:
    *   Run CARLA simulator.
    *   Execute `collect_carla_data.py` to record driving data (images and steering angles), saved into a structured directory (e.g., `datasets/YYYYMMDD_HHMMSS/images/` and `datasets/YYYYMMDD_HHMMSS/steering_data.csv`).
3.  **(Optional) Calculate Dataset Statistics**:
    *   Run `python calculate_dataset_std_mean.py --dataset_type <your_carla_dataset_folder_name>` to get mean/std. Update `config.py` or ensure normalization in `dataset_loader.py` is appropriate.
4.  **Model Training (`train.py`)**:
    *   Run `python train.py --dataset_types <your_carla_dataset_folder_name_1> <your_carla_dataset_folder_name_2> ...`
    *   The script loads data, trains the `NvidiaModel`, logs to TensorBoard, and saves the best model.
5.  **Model Validation (`validate.py`)**:
    *   After training, run `python validate.py` (ensure it's pointing to the correct model and dataset type, as modified earlier) to get performance metrics on a validation set.
6.  **Inference**:
    *   `inference_run_dataset.py`: Test the model on a collected CARLA dataset.
    *   `inference_run_video.py`: Record a video from CARLA and run inference on it.
7.  **Deployment (Optional)**:
    *   `export_model_to_onnx.py`: Convert the trained model to ONNX.

## 4. How to Run the Project (Focus on CARLA)

1.  **Install Dependencies**:
    *   `pip install -r requirements.txt`
    *   `pip install -r torchreq.txt --index-url https://download.pytorch.org/whl/cu118` (or appropriate for your CUDA version)
2.  **Setup CARLA Simulator**: Ensure CARLA is installed and running.
3.  **Collect Data**:
    *   `python capture_data/collect_carla_data.py --save-dir ./datasets --max-frames 1000` (adjust parameters as needed). This will create a new subfolder in `./datasets` like `YYYYMMDD_HHMMSS`.
4.  **Train the Model**:
    *   `python train.py --dataset_types YYYYMMDD_HHMMSS` (replace with your actual collected data folder name). You can list multiple dataset folders.
5.  **Monitor with TensorBoard**:
    *   `tensorboard --logdir=./runs` (or wherever `train.py` saves logs, check `config.py` or `train.py` for `log_dir`).
6.  **Validate**:
    *   Modify `validate.py` to use the correct dataset type (e.g., `dataset_types=['YYYYMMDD_HHMMSS']`) if it's different from 'carla_001'.
    *   Ensure `./save/model.pt` exists (or modify `validate.py` to load your specific model).
    *   `python validate.py`
7.  **Run Inference (Example on dataset)**:
    *   `python inference_run_dataset.py --model_path ./save/model.pt --dataset_type YYYYMMDD_HHMMSS`

## 5. Key Considerations & Potential Issues

*   **Dataset Paths and Structure**: Crucial. `collect_carla_data.py` now saves to `datasets/<timestamped_folder>/images` and `datasets/<timestamped_folder>/steering_data.csv`. Ensure `dataset_loader.py`'s `CarlaSimulatorDataset` is consistent with this (it expects `root_dir` to be `datasets/<timestamped_folder>`).
*   **Normalization**: Normalization to `[-1, 1]` occurs in `dataset_loader.py` during preprocessing. The `NvidiaModel` in `model.py` also has an initial normalization layer. This redundancy should ideally be resolved (likely keeping the one in `dataset_loader.py` and removing or ensuring the model's layer is an identity operation if data is already normalized).
*   **`dataset_type` Argument**: `train.py`, `validate.py`, etc., rely on `dataset_type` which should be the *name of the folder* inside `datasets/` (e.g., `20231027_103045`) when using `CarlaSimulatorDataset` as it's passed as the `root_dir` argument.
*   **Model Save/Load Paths**: Ensure consistency in where models are saved by `train.py` and loaded by `validate.py` and inference scripts (default is `./save/model.pt`).

This updated analysis provides a more current and detailed overview of the project components and flow. 