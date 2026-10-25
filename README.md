Link to the Streamlit app: [https://penguin-classification-eddie-0220.streamlit.app/](https://penguin-classification-eddie-0220.streamlit.app/)

# Penguin Species Classification

This project uses a **K-nearest-neighbors (KNN)** classifier to predict a penguin's species from its physical measurements and other characteristics. It also includes a Streamlit application that allows users to enter a new penguin's information and receive a predicted species and class probabilities.

The project uses the Palmer Penguins dataset, which contains observations of three species:

- Adelie
- Chinstrap
- Gentoo

## Project question

Can a penguin's species be predicted from its bill measurements, flipper length, body mass, island, sex, and year of observation?

## Features

The model uses the following predictor variables:

| Feature | Description |
| --- | --- |
| `bill_length_mm` | Bill length in millimeters |
| `bill_depth_mm` | Bill depth in millimeters |
| `flipper_length_mm` | Flipper length in millimeters |
| `body_mass_g` | Body mass in grams |
| `island` | Island where the penguin was observed |
| `sex` | Penguin sex |
| `year` | Year of observation |

The response variable is `species`.

## Methodology

The analysis follows these main steps:

1. Load the Palmer Penguins data from the public dataset URL.
2. Remove two observations with extensive missing information.
3. Encode categorical predictors.
4. Split the data into training and test sets.
5. Standardize the numerical predictors.
6. Use `KNNImputer` to handle missing values.
7. Use five-fold cross-validation on the training data to select the value of \(K\).
8. Evaluate the final model on the held-out test set.

The cross-validation error rate is calculated as:

\[
\text{Error rate} = 1 - \text{accuracy}
\]

The analysis notebook includes the exploratory analysis, preprocessing decisions, cross-validation results, error-rate plot, confusion matrix, classification report, and model interpretation.

## Streamlit application

The Streamlit app allows users to enter:

- Bill length
- Bill depth
- Flipper length
- Body mass
- Island
- Sex
- Year of observation

After the user selects **Predict**, the application displays:

- The predicted penguin species
- The selected value of \(K\)
- The predicted probability for each species
- A plot of cross-validation error rate versus \(K\)

## Run the application locally

Clone the repository and move into the project directory:

```bash
git clone https://github.com/EddieNguyen2012/penguin-classification.git
cd penguin-classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app1.py
```

The application should open in a new browser tab. The dataset is downloaded automatically from the Palmer Penguins GitHub repository when the app loads.

## Repository structure

```text
penguin-classification/
├── analysis.ipynb       # Exploratory analysis and model development
├── app1.py              # Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # Project license
```

## Dataset

The data are from the Palmer Penguins project:

- [Palmer Penguins project page](https://allisonhorst.github.io/palmerpenguins/)
- [Dataset repository](https://github.com/allisonhorst/palmerpenguins)
- [Direct CSV file](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv)

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Streamlit
