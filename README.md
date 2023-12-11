# Movie Recommendation System with Neural Collaborative Filtering

## Introduction

This Python script implements a movie recommendation system using Neural Collaborative Filtering. The recommendation model considers movie ratings and genres to provide personalized suggestions. The analysis involves data preprocessing, feature engineering, and building a neural network model using the Keras library.

## Usage

### Dependencies

Make sure you have the following libraries installed:

- pandas
- matplotlib
- numpy
- seaborn
- scikit-learn
- keras

You can install them using the following:

``bash
pip install pandas matplotlib numpy seaborn scikit-learn keras

# Movie Recommendation System with Neural Collaborative Filtering

## Introduction

This Python script implements a movie recommendation system using Neural Collaborative Filtering. The recommendation model considers movie ratings and genres to provide personalized suggestions. The analysis involves data preprocessing, feature engineering, and building a neural network model using the Keras library.

## Contents

- `movie_recommendation.py`: Python script containing the movie recommendation code.
- `ratings.csv`: CSV file containing movie ratings.
- `movies.csv`: CSV file containing movie information.

## Analysis Steps

### 1. Data Loading and Exploration:

- **Objective:** Load and understand the structure of the movie ratings and genres data.

- **Steps:**
  - Load movie ratings and genres data from CSV files.
  - Merge the datasets based on the movieId.
  - Display basic information about the merged dataset.

### 2. Data Preprocessing:

- **Objective:** Prepare the data for analysis and model training.

- **Steps:**
  - Handle missing values and unnecessary columns.
  - Extract the primary genre for each movie.

### 3. Mean Ratings Calculation:

- **Objective:** Calculate the weighted average rating for each movie based on its primary genre.

- **Steps:**
  - Group the data by movieId, title, and primary genre.
  - Calculate mean rating and rating count for each movie.
  - Merge the calculated ratings back into the main dataset.

### 4. Filtering Movies:

- **Objective:** Focus on movies with sufficient ratings and reasonable average ratings.

- **Steps:**
  - Filter out movies with fewer than 75 ratings.
  - Filter out movies with an average rating below 2.5.

### 5. Neural Collaborative Filtering:

- **Objective:** Train a neural network model for collaborative filtering.

- **Steps:**
  - Prepare features and target variables.
  - One-hot encode 'PrimaryGenre' and convert 'title' to integer labels.
  - Build and train a neural network model using Keras.

### 6. Model Evaluation:

- **Objective:** Assess the performance of the trained model.

- **Steps:**
  - Split the data into training and testing sets.
  - Evaluate the model on the test set using mean squared error.

## Usage

1. Clone the repository:

``bash
git clone https://github.com/your-username/movie-recommendation.git
cd movie-recommendation

