
import json
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Student answers and their labels
data = [
    # Loop boundary error
    ("I use less than or equal to n in my loop", "Loop boundary error"),
    ("My loop accesses one extra array position", "Loop boundary error"),
    ("I loop from zero through n inclusive", "Loop boundary error"),
    ("The last index of a list is n", "Loop boundary error"),
    ("I use range of n plus one to visit n items", "Loop boundary error"),
    ("My loop gives an index out of range error", "Loop boundary error"),
    ("I use i less than or equal to the list length", "Loop boundary error"),
    ("Using index n accesses beyond a list of length n", "Loop boundary error"),
    ("Using index equal to n causes an out of bounds error", "Loop boundary error"),
    ("I iterate one step beyond the last valid index", "Loop boundary error"),
    ("I stop at i less than n", "Correct concept"),
    ("The last valid index is length minus one", "Correct concept"),
    ("A list of length five has indices zero through four", "Correct concept"),
    ("A loop should stop before reaching the list length", "Correct concept"),
    ("Range n visits indices from zero to n minus one", "Correct concept"),
    ("For n elements, valid indices are zero to n minus one", "Correct concept"),
    ("A list of length ten has last index nine", "Correct concept"),

    # Binary search
    ("When the middle is too small I set low equal to mid", "Binary search boundary error"),
    ("Binary search repeats the same middle forever", "Binary search boundary error"),
    ("I change the wrong boundary after comparing the target", "Binary search boundary error"),
    ("When target is smaller than middle I increase low", "Binary search boundary error"),
    ("I never update the high or low boundary", "Binary search boundary error"),
    ("I set low to middle without adding one", "Binary search boundary error"),
    ("I forget to move the boundary after checking mid", "Binary search boundary error"),
    ("I keep mid inside the search range and repeat forever", "Binary search boundary error"),
    ("I update low to mid plus one", "Correct concept"),
    ("I reduce the search range after comparing the middle", "Correct concept"),
    ("Binary search discards the half that cannot contain the target", "Correct concept"),
    ("When the target is greater than mid, low becomes mid plus one", "Correct concept"),
    ("When target is bigger than mid, low becomes mid plus one", "Correct concept"),
    ("When target is smaller than mid, high becomes mid minus one", "Correct concept"),
    ("If target is greater than middle, move low to middle plus one", "Correct concept"),
    ("If target is smaller than middle, move high to middle minus one", "Correct concept"),

    # Stack order
    ("A stack follows first in first out", "Stack order confusion"),
    ("The oldest item is removed first from a stack", "Stack order confusion"),
    ("A stack removes the first inserted item first", "Stack order confusion"),
    ("Stack means FIFO", "Stack order confusion"),
    ("The first pushed element is always popped first", "Stack order confusion"),
    ("FIFO means the oldest item comes out first", "Stack order confusion"),
    ("The first inserted element comes out first in a stack", "Stack order confusion"),
    ("A stack follows last in first out", "Correct concept"),
    ("The newest pushed item is popped first", "Correct concept"),
    ("A stack follows LIFO order", "Correct concept"),
    ("A queue typically follows first in first out", "Correct concept"),
    ("A stack removes the most recently added item", "Correct concept"),
    ("The last item pushed is popped first", "Correct concept"),
    ("Stack removes the newest element first", "Correct concept"),
    ("Push adds an item to the top of a stack", "Correct concept"),
    ("Pop removes the top item from a stack", "Correct concept"),

    # Recursion
    ("My recursive function has no base case", "Missing recursion base case"),
    ("The function calls itself forever without stopping", "Missing recursion base case"),
    ("I forgot the stopping condition in recursion", "Missing recursion base case"),
    ("My recursive calls never get closer to the stopping condition", "Missing recursion base case"),
    ("I keep calling the same function without a stopping condition", "Missing recursion base case"),
    ("Without a base case recursion may never stop", "Missing recursion base case"),
    ("My recursive function has a base case at zero", "Correct concept"),
    ("The base case stops more recursive calls", "Correct concept"),
    ("Every recursive call should move toward the base case", "Correct concept"),
    ("A recursive function must eventually reach its base case", "Correct concept"),
    ("The base case defines when recursion ends", "Correct concept"),
    ("Base case stops recursive calls", "Correct concept"),
    ("The loop runs one extra time", "Loop boundary error"),
    ("The loop stops too early", "Loop boundary error"),
    ("The stack removes the oldest item first", "Stack order confusion"),
    ("The recursive function never stops", "Missing recursion base case"),
    ("The answer is correct", "Correct concept"),
    ("My loop accesses the last invalid index", "Loop boundary error"),
("I use i <= len(array)", "Loop boundary error"),
("Binary search low should become mid + 1", "Correct concept"),
("I forget to update high in binary search", "Binary search boundary error"),
("A stack removes the newest item first", "Correct concept"),
("A stack removes the oldest item first", "Stack order confusion"),
("Recursion needs a stopping condition", "Correct concept"),
("My recursive function never reaches a base case", "Missing recursion base case"),
("A list index starts at zero", "Correct concept"),
("The loop repeats one extra iteration", "Loop boundary error"),

    # Additional Loop boundary examples
    ("I use i <= len(items) and get an index error", "Loop boundary error"),
    ("The loop goes one position past the last item", "Loop boundary error"),
    ("I use range(len(items) + 1)", "Loop boundary error"),
    ("The last index is len(items)", "Loop boundary error"),
    ("I access items[len(items)]", "Loop boundary error"),
    ("I use i < len(items) to visit each item", "Correct concept"),
    ("range(len(items)) visits valid list indices", "Correct concept"),
    ("The last index is len(items) - 1", "Correct concept"),

    # Additional Binary search examples
    ("I search the entire list one by one", "Binary search boundary error"),
    ("I forget to update the low pointer", "Binary search boundary error"),
    ("I update high to mid instead of mid minus one", "Binary search boundary error"),
    ("Binary search works on sorted data", "Correct concept"),
    ("I compare the target with the middle element", "Correct concept"),
    ("I update the search range after each comparison", "Correct concept"),

    # Additional Stack examples
    ("I remove the first item pushed onto the stack", "Stack order confusion"),
    ("I think a stack follows FIFO order", "Stack order confusion"),
    ("I pop the most recently pushed item", "Correct concept"),
    ("A stack follows last in first out", "Correct concept"),
    ("The last item pushed is removed first", "Correct concept"),

    # Additional Recursion examples
    ("My recursive function has no stopping condition", "Missing recursion base case"),
    ("The function keeps calling itself forever", "Missing recursion base case"),
    ("I forgot to define the base case", "Missing recursion base case"),
    ("The base case stops further recursive calls", "Correct concept"),
    ("Every recursive call should move toward the base case", "Correct concept"),

]

X = [item[0] for item in data]
y = [item[1] for item in data]


def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000
        )),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.5
        ))
    ])


# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train evaluation model
evaluation_model = create_model()
evaluation_model.fit(X_train, y_train)
predictions = evaluation_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)
# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=evaluation_model.classes_,
)

print("\nConfusion Matrix:")
print("Class labels:", evaluation_model.classes_)
print(cm)

print("\n========== MODEL EVALUATION ==========")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")
print("\nPrecision, Recall and F1-score:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save evaluation metrics
root = Path(__file__).parent
metrics_path = root / "model_metrics.json"

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision_macro": round(report["macro avg"]["precision"] * 100, 2),
    "recall_macro": round(report["macro avg"]["recall"] * 100, 2),
    "f1_macro": round(report["macro avg"]["f1-score"] * 100, 2),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "dataset_size": len(X),
    "note": "Small demonstration dataset; results are preliminary.",
    "confusion_matrix": cm.tolist(),
    "classes":evaluation_model.classes_.tolist(),
}

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# Train final model on all examples
final_model = create_model()
final_model.fit(X, y)

model_path = root / "errorlens_model.joblib"
joblib.dump(final_model, model_path)

print("\n========== SAVED FILES ==========")
print(f"Model saved: {model_path}")
print(f"Metrics saved: {metrics_path}")
print(f"Total examples: {len(X)}")
print(f"Categories: {len(set(y))}")
print("Training completed successfully.")
