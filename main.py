import os
import src.load_data
import src.model
import src.preprocessing



def main():
    """
    Arbeitsplan:
    - Daten laden
    - Daten aufbereiten
    - Modell trainieren
    - Modell evaluieren
    """
    # Daten laden
    data = src.load_data.load_data()

    # Daten aufbereiten
    processed_data = src.preprocessing.preprocess_data()

    # Modell trainieren
    model = src.model.train_model()

    # Modell evaluieren
    evaluation_results = src.model.evaluate_model()

    print("Evaluation Results:", evaluation_results)


if __name__ == "__main__":
    main()
