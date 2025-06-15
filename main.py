import os
import datetime
import torch
from src.load_data import load_data
from src.model import MLPClassifier, train, evaluate, save_model
from src.preprocessing import preprocess_data

# Define constants
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIRECTORY = f"saved_models/model_{TIMESTAMP}"
MODELPATH = f"saved_models/model_{TIMESTAMP}/best_model.pth"
HYPERS_PATH = f"saved_models/model_{TIMESTAMP}/hyperparameters.txt"

# Define Hyperparameters
HIDDEN_DIM = 256
OUTPUT_DIM = 2
DROPOUT = 0.5
EPOCHS = 50
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
NUM_HIDDEN_LAYERS = 20

def main():
    """
    main function to train the model
    - set device
    - load data
    - preprocess data
    - train model + evaluate
    """
    # create directory for models
    os.makedirs(DIRECTORY, exist_ok=True)
    
    # save hyperparameters
    save_hyperparameters()

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # load data
    data = load_data()

    # preprocess data
    train_data, test_data = preprocess_data(data, BATCH_SIZE)

    # set input dimension with number of features
    INPUT_DIM = next(iter(train_data))[0].shape[1]

    # initialize model
    model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, DROPOUT, OUTPUT_DIM, NUM_HIDDEN_LAYERS).to(device)

    # train model + evaluate
    model = train(model, train_data, test_data, MODELPATH, EPOCHS, LEARNING_RATE, device)


def save_hyperparameters():
    """
    Save hyperparameters to file
    """
    hyperparameters = {
        "HIDDEN_DIM": HIDDEN_DIM,
        "OUTPUT_DIM": OUTPUT_DIM,
        "DROPOUT": DROPOUT,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS
    }
    
    with open(HYPERS_PATH, 'w') as f:
        f.write("Model Hyperparameters:\n")
        f.write("=====================\n\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
