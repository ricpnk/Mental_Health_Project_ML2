# Machine Learning Project: Mental Health

This project is intended for academic purposes as part of the Machine Learning 2 module at Hochschule Landshut.

## Description
This project aims to develop a machine learning model to predict mental health outcomes based on a dataset provided by Kaggle. The dataset contains various features related to mental health, and the goal is to build a model that can accurately classify or predict mental health conditions.

### Link to the dataset
https://www.kaggle.com/competitions/playground-series-s4e11/overview



## Important Notes

- 25 minutes presentation (not longer!)
- include some recap of the theory you are using in your projects
- project presentations: last 3 weeks of June during our regular time slots, 3 presentations each time, i.e. 6/week.
- some of the projects are from Kaggle competitions. Here your result will be compared to the results of the entered projects.



## Project Structure

- `src/`: Source code directory
  - `models.py`: Contains the model architecture (Encoder, Decoder, Attention, Seq2Seq)
  - `load_data.py`: Data loading
  - `preprocessing.py`: Data preprocessing
- `data/`: Directory containing the mental health dataset
- `notebooks/`: Jupyter notebook for data exploration
- `saved_models/`: Directory for storing model checkpoints
- `main.py`: Main script to run the project
- `pyproject.toml`: Project metadata and dependencies
- `README.md`: Project description and instructions

## How to Run

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies and execute scripts in the virtual environment.

### Install Dependencies

If you haven't already, install `uv`:

```bash
pip install uv
```

Then install all project dependencies:

```bash
uv sync
```

### Run the Scripts

To execute any of the scripts, use:

```bash
uv run <script.py>
```

For example:

```bash
uv run main.py
```

## File Descriptions

### `main.py`
The main script that handles:
- Data loading
- Data preprocessing
- Model training and evaluation

### `src/load_data.py`
- implement me........

### `src/preprocessing.py`
- implement me........

### `src/models.py`
- implement me........

### `notebooks/data_exploration.ipynb`
- implement me........

