
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import random


# ================= CONFIGURATION =================
st.set_page_config(page_title="ErrorLens AI", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "errorlens_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
ATTEMPTS_PATH = BASE_DIR / "attempts.csv"
CSS_PATH = BASE_DIR / "style.css"

st.title("ErrorLens AI")
st.caption("Understand student misconceptions and practise concepts.")

if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ================= QUESTION BANK =================
QUESTIONS = {
    "Loops and array indices": [
        {
            "question": "What does range(4) produce?",
            "accepted": ["0 1 2 3", "0, 1, 2, 3", "range(4)"],
            "hint": "Python starts counting from 0.",
            "explanation": "range(4) produces 0, 1, 2, 3.",
            "simple_question": "What does range(3) produce?",
            "simple_answer": ["0 1 2", "0, 1, 2"],
            "category": "Loop boundary error",
        },
        {
            "question": "What is the last index of a list of length 5?",
            "accepted": ["4", "four", "length - 1", "n - 1"],
            "hint": "List indices start at 0.",
            "explanation": "The last index is length minus one.",
            "simple_question": "What is the last index of a list of length 3?",
            "simple_answer": ["2", "two"],
            "category": "Loop boundary error",
        },
        {
            "question": "What does range(1, 5) produce?",
            "accepted": ["1 2 3 4", "1, 2, 3, 4"],
            "hint": "The stop value is excluded.",
            "explanation": "range(1, 5) produces 1, 2, 3, 4.",
            "simple_question": "Does range(1, 3) include 3?",
            "simple_answer": ["no", "false"],
            "category": "Loop boundary error",
        },
        {
            "question": "How do you access the first item in list items?",
            "accepted": ["items[0]", "items [0]"],
            "hint": "The first index is zero.",
            "explanation": "Use items[0] to access the first item.",
            "simple_question": "What is the first index in Python?",
            "simple_answer": ["0", "zero"],
            "category": "Loop boundary error",
        },
        {
            "question": "How can you print every item in a list named items?",
            "accepted": [
                "for item in items: print(item)",
                "for item in items print(item)",
            ],
            "hint": "Loop directly through the list.",
            "explanation": "A for loop can visit each item and print it.",
            "simple_question": "Which keyword starts a Python for loop?",
            "simple_answer": ["for"],
            "category": "Loop boundary error",
        },
    ],

    "Binary search": [
        {
            "question": "In [2, 4, 6, 8, 10], what middle value is checked first?",
            "accepted": ["6", "six"],
            "hint": "Find the middle element.",
            "explanation": "The middle value is 6.",
            "simple_question": "What is the middle value in [1, 3, 5]?",
            "simple_answer": ["3", "three"],
            "category": "Binary search boundary error",
        },
        {
            "question": "Must the list be sorted for standard binary search?",
            "accepted": ["yes", "yes it must", "true"],
            "hint": "Binary search eliminates half the search range.",
            "explanation": "Standard binary search requires sorted data.",
            "simple_question": "Should binary search data be sorted?",
            "simple_answer": ["yes"],
            "category": "Binary search boundary error",
        },
        {
            "question": "What does binary search compare first?",
            "accepted": ["middle element", "middle", "mid", "middle value"],
            "hint": "It begins at the centre.",
            "explanation": "It compares the target with the middle element.",
            "simple_question": "Does binary search start at the middle?",
            "simple_answer": ["yes"],
            "category": "Binary search boundary error",
        },
        {
            "question": "If the target is smaller than the middle value, which half is searched?",
            "accepted": ["left half", "left", "lower half"],
            "hint": "Smaller values are on the left in sorted data.",
            "explanation": "Search the left half.",
            "simple_question": "Smaller values are on which side of a sorted list?",
            "simple_answer": ["left", "left side"],
            "category": "Binary search boundary error",
        },
        {
            "question": "What is the time complexity of binary search?",
            "accepted": ["o(log n)", "log n", "logarithmic", "ologn"],
            "hint": "The search range is halved each step.",
            "explanation": "Binary search has O(log n) time complexity.",
            "simple_question": "Does binary search halve the search range?",
            "simple_answer": ["yes"],
            "category": "Binary search boundary error",
        },
    ],

    "Stacks": [
        {
            "question": "Push A and then B onto an empty stack. Which is popped first?",
            "accepted": ["b"],
            "hint": "Remember LIFO.",
            "explanation": "B was pushed last, so it is popped first.",
            "simple_question": "Push 1 and then 2. Which is popped first?",
            "simple_answer": ["2", "two"],
            "category": "Stack order confusion",
        },
        {
            "question": "What does LIFO stand for?",
            "accepted": ["last in first out", "last in, first out"],
            "hint": "The newest item leaves first.",
            "explanation": "LIFO means Last In, First Out.",
            "simple_question": "Does a stack remove the newest item first?",
            "simple_answer": ["yes"],
            "category": "Stack order confusion",
        },
        {
            "question": "Which operation adds an item to a stack?",
            "accepted": ["push"],
            "hint": "The opposite operation is pop.",
            "explanation": "Push adds an item to the stack.",
            "simple_question": "Is push used to add an item?",
            "simple_answer": ["yes"],
            "category": "Stack order confusion",
        },
        {
            "question": "Which operation removes the top item from a stack?",
            "accepted": ["pop"],
            "hint": "It removes the most recently added item.",
            "explanation": "Pop removes the top item.",
            "simple_question": "Does pop remove an item?",
            "simple_answer": ["yes"],
            "category": "Stack order confusion",
        },
        {
            "question": "Which item is at the top after pushing 4, then 7?",
            "accepted": ["7", "seven"],
            "hint": "The last pushed item is on top.",
            "explanation": "7 is at the top of the stack.",
            "simple_question": "After pushing A then B, what is on top?",
            "simple_answer": ["b"],
            "category": "Stack order confusion",
        },
    ],

    "Recursion": [
        {
            "question": "What is the purpose of a base case in recursion?",
            "accepted": [
                "to stop recursion", "stop recursion",
                "stopping condition", "termination condition", "base case",
            ],
            "hint": "What tells the function to stop?",
            "explanation": "A base case stops further recursive calls.",
            "simple_question": "What stops recursion?",
            "simple_answer": ["base case", "stopping condition"],
            "category": "Missing recursion base case",
        },
        {
            "question": "What happens if recursion has no stopping condition?",
            "accepted": [
                "infinite recursion", "infinite calls",
                "stack overflow", "runs forever",
            ],
            "hint": "Think about a function calling itself repeatedly.",
            "explanation": "It may continue until a recursion limit or stack error occurs.",
            "simple_question": "Can recursion without a base case cause an error?",
            "simple_answer": ["yes"],
            "category": "Missing recursion base case",
        },
        {
            "question": "What is a recursive function?",
            "accepted": [
                "a function that calls itself",
                "function that calls itself",
                "calls itself",
            ],
            "hint": "The function repeats by calling itself.",
            "explanation": "A recursive function calls itself.",
            "simple_question": "Does a recursive function call itself?",
            "simple_answer": ["yes"],
            "category": "Missing recursion base case",
        },
        {
            "question": "What is factorial of 3?",
            "accepted": ["6", "six"],
            "hint": "Calculate 3 × 2 × 1.",
            "explanation": "3 factorial equals 3 × 2 × 1 = 6.",
            "simple_question": "What is 2 factorial?",
            "simple_answer": ["2", "two"],
            "category": "Missing recursion base case",
        },
        {
            "question": "What is the base case for factorial when n equals 0?",
            "accepted": ["1", "one", "return 1"],
            "hint": "Zero factorial has a defined value.",
            "explanation": "The base case returns 1 when n is 0.",
            "simple_question": "What is 0 factorial?",
            "simple_answer": ["1", "one"],
            "category": "Missing recursion base case",
        },
    ],
}

CATEGORY_EXPLANATIONS = {
    "Loop boundary error": (
        "Check Python indexing and whether range() excludes its stop value."
    ),
    "Binary search boundary error": (
        "Check the middle, low and high positions. Data must be sorted."
    ),
    "Stack order confusion": "A stack follows Last In, First Out (LIFO).",
    "Missing recursion base case": (
        "A recursive function needs a stopping condition called a base case."
    ),
    "Correct concept": "Your answer matches the expected concept.",
    "Uncertain": "Review the hint and explanation, then try again.",
} 

# ================= HELPER FUNCTIONS =================
def normalise(value):
    return " ".join(str(value).strip().lower().split())


def is_correct_answer(answer, topic, accepted):
    normalised_answer = normalise(answer)

    # Check normal accepted answers first.
    if normalised_answer in accepted:
        return True

    # Recognise a valid list-traversal code example.
    if topic == "Loops and array indices":
        code = "".join(normalised_answer.split())

        has_loop = (
            "foriinrange(len(items)):" in code
            or "foriinrange(4):" in code
        )
        has_print = (
            "print(items[i])" in code
            or "print(items[i])" in code
        )

        if has_loop and has_print:
            return True

    return False


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


model = load_model()


def save_attempt(record):
    new_data = pd.DataFrame([record])

    if ATTEMPTS_PATH.exists():
        try:
            old_data = pd.read_csv(ATTEMPTS_PATH)
            new_data = pd.concat(
                [old_data, new_data],
                ignore_index=True,
            )
        except Exception:
            pass

    new_data.to_csv(ATTEMPTS_PATH, index=False)


def classify_answer(answer, topic, correct):
    if correct:
        return "Correct concept", 1.0, "Answer check"

    if model is not None:
        try:
            label = str(model.predict([answer])[0])
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([answer])[0]
                confidence = float(max(probabilities))

                if confidence < 0.40:
                    return (
                        QUESTIONS[topic]["category"],
                        confidence,
                        "Topic-based fallback",
                    )

            return label, confidence, "ML model"

        except Exception:
            pass

    return QUESTIONS[topic][0]["category"],None,"Topic-based fallback"


def get_correct_values(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ================= NAVIGATION =================
page = st.sidebar.radio(
    "Navigation",
    ["Student Practice", "Teacher Dashboard"],
    index=0,
)


# ================= STUDENT PRACTICE =================
if page == "Student Practice":
    st.header("Student Practice")
    st.write(
        "Choose a topic, answer the question, and review the feedback."
    )

    student_name = st.text_input("Student name (optional)")
    topic = st.selectbox("Choose a topic", list(QUESTIONS.keys()))
    question =random.choice(QUESTIONS[topic])

    st.info(question["question"])

    answer = st.text_area(
        "Enter your answer and reasoning",
        placeholder="Explain how you arrived at your answer...",
        height=130,
    )

    if st.button(
        "Analyse My Answer",
        type="primary",
        width="stretch",
    ):
        if not answer.strip():
            st.warning("Please enter your answer first.")
        else:
            accepted = [
                normalise(item)
                for item in question["accepted"]
            ]

            correct = is_correct_answer(
                answer,
                topic,
                accepted,
            )

            label, confidence, method = classify_answer(
                answer,
                topic,
                correct,
            )

            record = {
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "student_name": student_name.strip() or "Anonymous",
                "topic": topic,
                "answer": answer.strip(),
                "predicted_label": label,
                "is_correct": correct,
                "prediction_method": method,
                "confidence": confidence,
            }

            save_attempt(record)
            st.session_state["last_result"] = record
            st.session_state["last_topic"] = topic

    result = st.session_state.get("last_result")

    if result and st.session_state.get("last_topic") == topic:
        st.markdown("---")
        st.subheader("Analysis Result")

        if result["is_correct"]:
            st.success("Your answer looks correct.")
        else:
            st.warning("Your answer may need another look.")

        st.write("**Likely category:**", result["predicted_label"])

        explanation = CATEGORY_EXPLANATIONS.get(
            result["predicted_label"],
            CATEGORY_EXPLANATIONS.get(
                question["category"],
                "Review the concept and try again.",
            ),
        )
        st.write(explanation)
        st.caption(f"Analysis method: {result['prediction_method']}")

        if result.get("confidence") is not None:
            st.caption(
                f"Model confidence: {result['confidence']:.1%}. "
                "This is an estimate, not a guarantee."
            )

        with st.expander("Show hint"):
            st.write(question["hint"])

        with st.expander("See explanation"):
            st.write(question["explanation"])

        if not result["is_correct"]:
            with st.expander("Try a simpler question"):
                st.write(question["simple_question"])

                simple_answer = st.text_input(
                    "Your answer",
                    key=f"simple_answer_{topic}",
                )

                if st.button(
                    "Check simpler answer",
                    key=f"simple_check_{topic}",
                ):
                    valid_answers = [
                        normalise(item)
                        for item in question["simple_answer"]
                    ]

                    if normalise(simple_answer) in valid_answers:
                        st.success(
                            "Correct. Now try the original question again."
                        )
                    else:
                        st.info(
                            "Not quite. Review the hint and try again."
                        )


# ================= TEACHER DASHBOARD =================
else:
    st.header("Teacher Dashboard")

    if ATTEMPTS_PATH.exists():
        try:
            attempts = pd.read_csv(ATTEMPTS_PATH)
        except Exception:
            attempts = pd.DataFrame()
    else:
        attempts = pd.DataFrame()

    if attempts.empty:
        st.info(
            "No submissions yet. Complete a question in Student Practice first."
        )
    else:
        total = len(attempts)

        if "is_correct" in attempts.columns:
            correct_values = get_correct_values(
                attempts["is_correct"]
            )
            correct_count = int(correct_values.sum())
        else:
            correct_values = pd.Series(
                False,
                index=attempts.index,
            )
            correct_count = 0

        topic_count = (
            attempts["topic"].nunique()
            if "topic" in attempts.columns
            else 0
        )

        misconception_labels = [
            "Loop boundary error",
            "Binary search boundary error",
            "Stack order confusion",
            "Missing recursion base case",
            "Uncertain",
        ]

        if "predicted_label" in attempts.columns:
            potential_count = int(
                attempts["predicted_label"]
                .isin(misconception_labels)
                .sum()
            )
        else:
            potential_count = 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Submissions", total)
        col2.metric("Correct Answers", correct_count)
        col3.metric("Topics Explored", topic_count)
        col4.metric("Potential Misconceptions", potential_count)

        if "predicted_label" in attempts.columns:
            st.subheader("Most Common Categories")
            category_counts = attempts["predicted_label"].value_counts()
            st.bar_chart(category_counts)

        st.subheader("Topics Students Struggle With")

        if (
            "is_correct" in attempts.columns
            and "topic" in attempts.columns
        ):
            wrong_answers = attempts[~correct_values]

            if not wrong_answers.empty:
                topic_errors = wrong_answers["topic"].value_counts()
                st.bar_chart(topic_errors)
                st.caption(
                    "Each bar shows the number of incorrect answers per topic."
                )
            else:
                st.info("No incorrect answers recorded yet.")
        else:
            st.info("Required answer data is not available.")

        st.subheader("Student Submissions")
        st.dataframe(
            attempts.tail(20),
            width="stretch",
        )

        st.download_button(
            "Download CSV",
            data=attempts.to_csv(index=False).encode("utf-8"),
            file_name="attempts.csv",
            mime="text/csv",
            width="stretch",
        )

    # ================= MODEL EVALUATION =================
    st.markdown("---")
    st.subheader("Model Evaluation")

    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
                
            if "confusion_matrix" in metrics and "classes" in metrics:
                st.subheader("Confusion Matrix")
                cm_df = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["classes"],
                    columns=metrics["classes"],
                )
                st.dataframe(cm_df, width="stretch")
                st.caption(
                    "Rows = actual classes; columns = predicted classes."
                )
            metric_names = {
                "accuracy": "Accuracy",
                "precision_macro": "Precision",
                "recall_macro": "Recall",
                "f1_macro": "F1-score",
            }

            available_metrics = [
                (key, label)
                for key, label in metric_names.items()
                if isinstance(metrics.get(key), (int, float))
            ]

            if available_metrics:
                metric_columns = st.columns(len(available_metrics))

                for index, (key, label) in enumerate(available_metrics):
                    value = float(metrics[key])

                    if value > 1:
                        value = value / 100

                    metric_columns[index].metric(
                        label,
                        f"{value:.2%}",
                    )
            else:
                st.info(
                    "No recognised metrics found in model_metrics.json."
                )

            st.caption(
                "Small demonstration dataset: these evaluation results "
                "are preliminary and may change with more labelled examples."
            )

        except (json.JSONDecodeError, OSError, TypeError):
            st.warning(
                "Could not read model_metrics.json. "
                "Check that it contains valid JSON."
            )
    else:
        st.info(
            "model_metrics.json not found. Run train_model.py to generate metrics."
        )


# ================= SIDEBAR FOOTER =================
st.sidebar.markdown("---")
st.sidebar.caption("ErrorLens AI")