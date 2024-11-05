# Restaurant Review Predictor
## Project Overview

The **Restaurant Review Predictor** is a machine learning project that classifies restaurant reviews as positive or negative. Using Natural Language Processing (NLP) techniques, this project employs text preprocessing, feature extraction, and a logistic regression model to analyze and predict sentiment from customer reviews. This tool can help restaurant owners understand customer feedback and improve their services.

![Project Overview](https://sevenrooms.com/_next/image/?url=https%3A%2F%2Fcdn.builder.io%2Fapi%2Fv1%2Fimage%2Fassets%252Facabda4a1104467f93cbad15a924c9a3%252F436f13946b344aaf9d7ab7d505e1d669&w=1920&q=75)

## Features

- **Sentiment Analysis**: Classifies reviews into positive and negative categories.
- **Text Preprocessing**: Cleans and prepares text data by removing special characters and stop words.
- **Feature Extraction**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical format suitable for machine learning algorithms.
- **Model Training and Evaluation**: Implements logistic regression for model training and provides evaluation metrics such as precision, recall, and F1-score.
- **User-Friendly Prediction**: Allows users to input their own reviews for sentiment prediction.

## Installation

To run this project, ensure you have Python installed (preferably version 3.6 or higher). You will also need to install the required libraries. Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/restaurant-review-predictor.git
   cd restaurant-review-predictor
   ```
   Install the required packages:

bash
Copy code
pip install pandas nltk scikit-learn
Download the NLTK resources: In a Python shell, run the following commands to download the necessary NLTK data:

python
Copy code
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
Usage
Prepare the Dataset: Ensure the dataset Restaurant_Reviews.tsv is in the project directory. This dataset should contain two columns: "Review" and "Liked."

Run the Code: You can execute the script using Jupyter Notebook or a Python script. Ensure that all libraries are correctly installed.

Input Reviews for Prediction: Modify the variable test in the code to include the review you want to analyze. The prediction will classify the review as positive or negative.

python
Copy code
test = "Your review here."
Code Explanation
Data Loading: The dataset is loaded using pandas from a TSV file.

python
Copy code
df = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')
Text Preprocessing: The reviews are cleaned using regex, converted to lowercase, and stemmed to reduce words to their root form using PorterStemmer.

python
Copy code
Review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
TF-IDF Vectorization: The text data is transformed into numerical vectors using the TfidfVectorizer.

python
Copy code
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
Model Training: A logistic regression model is trained using the training data.

python
Copy code
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train_vect, y_train)
Prediction: New reviews can be inputted for sentiment prediction.

python
Copy code
prediction = clf.predict(example_counts)
Results
After running the model, the following classification report is generated, which includes metrics like precision, recall, and F1-score. Here's an example of what the output might look like:

markdown
Copy code
precision recall f1-score support

           0       0.75      0.85      0.80       143
           1       0.84      0.75      0.79       157

    accuracy                           0.79       300

macro avg 0.80 0.80 0.79 300
weighted avg 0.80 0.79 0.79 300
The accuracy of the model indicates that it can effectively predict sentiment in restaurant reviews.

Screenshots
Here are some results of the project in action:

This is a positive Review

This is a Negative review

(Replace the above image paths with actual images that you have in your project.)
