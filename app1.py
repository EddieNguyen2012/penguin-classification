import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Page title
# -----------------------------

st.set_page_config(
    page_title="KNN Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Predicting penguin species using K-nearest-neighbors")
st.write(
    """
    This app uses a K-nearest-neighbors (KNN) classifier to predict the species of penguins based on their physical characteristics. The model is trained on a dataset of penguin measurements
    and can be used to make predictions for new observations.
    """
)

# -----------------------------
# Load and prepare the data
# -----------------------------

@st.cache_data
def load_data():
    data_url = (
        "https://raw.githubusercontent.com/allisonhorst/"
        "palmerpenguins/main/inst/extdata/penguins.csv"
    )
    df = pd.read_csv(data_url)
    df.drop([3, 271], axis=0, inplace=True) # drop rows with missing values found during EDA
    return df


df = load_data()

numerical_features = [col for col in df.columns if not df[col].dtype in ['str', 'object']]

column_transformer = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(drop='first'), ['island']),
        ('label_encoder', OrdinalEncoder(), ['sex']),
        ('scaler', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

preprocess = Pipeline(steps=[
    ('column_transformer', column_transformer),
    ('imputer', KNNImputer(n_neighbors=5)),
])

y_encoded = OrdinalEncoder().fit_transform(df[['species']])

df = df.drop('species', axis=1) # drop the target variable from the features dataframe

x_train, x_test, y_train, y_test = train_test_split(df, y_encoded.ravel(), test_size=0.2, random_state=42)

X_train = preprocess.fit_transform(x_train)
X_test = preprocess.transform(x_test)

# -----------------------------
# App tab
# -----------------------------

[tab] = st.tabs(
    ["Make a Prediction"]
)

# -----------------------------
# Select the best value of k
# -----------------------------

k_values = range(1, 21) 

knn_pipeline = Pipeline(steps=[
    ('knn', KNeighborsClassifier())
])

grid_search = GridSearchCV(knn_pipeline, param_grid={'knn__n_neighbors': k_values}, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

best_k = grid_search.best_params_['knn__n_neighbors']
# -----------------------------
# Tab : Make a prediction
# -----------------------------

with tab:
    st.header("Predict for a new penguin")

    st.write(
        """
        Enter characteristics for a new penguin. The model will predict the species of the penguin based on the input features.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        bill_length = st.number_input(
            "Bill length (mm)",
            min_value=30.0,
            max_value=60.0,
            value=45.0
        )

        bill_depth = st.number_input(
            "Bill depth (mm)",
            min_value=10.0,
            max_value=25.0,
            value=15.0
        )

        flipper_length = st.number_input(
            "Flipper length (mm)",
            min_value=150.0,
            max_value=250.0,
            value=200.0
        )

        body_mass = st.number_input(
            "Body mass (g)",
            min_value=2500.0,
            max_value=6500.0,
            value=4500.0
        )

    with col2:
        island = st.selectbox(
            "Island",
            ["Torgersen", "Biscoe", "Dream"]
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        year = st.number_input(
            "Year",
            min_value=2007,
            max_value=2009,
            value=2007)

    if st.button("Predict"):
        new_penguin = pd.DataFrame({
            "bill_length_mm": [bill_length],
            "bill_depth_mm": [bill_depth],
            "flipper_length_mm": [flipper_length],
            "body_mass_g": [body_mass],
            "island": [island],
            "sex": [sex],
            "year": [2007]
        })
        new_penguin = preprocess.transform(new_penguin)
        prediction = model.predict(new_penguin)[0]
        probabilities = model.predict_proba(new_penguin)[0]

        class_mapping = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
        predicted_species = class_mapping[prediction]
        st.info(f"The model predicts the species of the penguin as: {predicted_species} with K = {best_k}")

        probability_df = pd.DataFrame({
            "Class": ["Adelie", "Chinstrap", "Gentoo"],
            "Predicted probability": probabilities
        })

        st.dataframe(probability_df)

        error_rates = 1- grid_search.cv_results_['mean_test_score']

        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(1, 21), error_rates, marker='o', linestyle='--')
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K Value')
        plt.ylabel('Error Rate')

        st.pyplot(fig)