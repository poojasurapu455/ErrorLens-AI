<<<<<<< HEAD
# ErrorLens AI
### Student Misconception Detector

## Problem Statement
Students often make mistakes because of conceptual misunderstandings. Traditional systems may identify whether an answer is correct or incorrect without explaining the likely reason.

## Solution
ErrorLens AI is a Streamlit-based prototype that analyses student answers, predicts likely misconception categories, and provides explanations, hints, and simpler practice questions.

## Features
- Student Practice interface
- Likely misconception classification
- Hints and concept explanations
- Simpler practice questions
- Teacher Dashboard
- Topic-wise incorrect answer charts
- Submission history and CSV export
- Model evaluation metrics

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## How to Run
1. Install the required packages:
   `pip install -r requirements.txt`
2. Train the model:
   `python train_model.py`
3. Start the application:
   `streamlit run app.py`

## Dataset and Evaluation
The current demonstration dataset contains 61 examples. The recorded test accuracy is 57.89% on 19 test examples.

These results are preliminary and do not establish reliable performance on unseen real-world student answers.

## Future Scope
- Expand and improve the labelled dataset
- Improve model accuracy and generalisation
- Test with more student responses
- Add more subjects and misconception categories
=======
# ErrorLens-AI
>>>>>>> 3467fd8fd26335baac3ed2d7a88d1f9a904ea85d
