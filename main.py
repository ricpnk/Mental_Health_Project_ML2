import os
import datetime
from src.load_data import load_data
from src.model import MLPClassifier, train_model, evaluate_model, save_model
from src.preprocessing import preprocess_data

# Define constants
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODELPATH = f"saved_models/model{TIMESTAMP}.pth"

# Define Hyperparameters
INPUT_DIM = 0
HIDDEN_DIM = 0
OUTPUT_DIM = 0
EPOCHS = 0
LEARNING_RATE = 0
BATCH_SIZE = 0



def main():
    """
    Arbeitsplan:
    - Daten laden
    - Daten aufbereiten
    - Modell trainieren
    - Modell evaluieren
    """
    # Sicherstellen, dass der Ordner für Modelle existieren
    os.makedirs("saved_models", exist_ok=True)
    
    # Daten laden
    train_data, test_data = load_data()

    # Daten aufbereiten
    processed_train_data, processed_test_data = preprocess_data(train_data, test_data)

    # Model initialisieren
    model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # Modell trainieren
    train_model(model, processed_train_data)
    save_model(model, MODELPATH)
    print("Model saved as trained_model.pth")

    # Modell evaluieren
    evaluation_results = evaluate_model(processed_test_data, model)

    print("Evaluation Results:", evaluation_results)


if __name__ == "__main__":
    main()
