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

# Define Hyperparameters
HIDDEN_DIM = 256
OUTPUT_DIM = 2
DROPOUT = 0.4
EPOCHS = 50
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
NUM_HIDDEN_LAYERS = 5



def main():
    """
    Arbeitsplan:
    - Daten laden
    - Daten aufbereiten
    - Modell trainieren
    - Modell evaluieren
    """
    # Sicherstellen, dass der Ordner für Modelle existieren
    os.makedirs(DIRECTORY, exist_ok=True)

    # Setzen des Geräts (CPU oder GPU)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Daten laden
    data = load_data()

    # Daten aufbereiten
    train_data, test_data = preprocess_data(data, BATCH_SIZE)

    # set input dimension with number of features
    INPUT_DIM = next(iter(train_data))[0].shape[1]

    # Model initialisieren
    model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, DROPOUT, OUTPUT_DIM, NUM_HIDDEN_LAYERS).to(device)

    # Modell trainieren
    model, accuracy = train(model, train_data, test_data, MODELPATH, EPOCHS, LEARNING_RATE, device)

    # Modell evaluieren
    print("Evaluation Results:", accuracy)


if __name__ == "__main__":
    main()
