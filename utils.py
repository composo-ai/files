import json
import aiohttp
import numpy as np
from typing import List, Dict, Optional
import asyncio

from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(results: dict):
    # Set the style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create DataFrame
    data = []
    for model, (correct, equal, incorrect) in results.items():
        data.append({"Model": model, "Correct": correct, "Ambiguous": equal, "Incorrect": incorrect})

    df = pd.DataFrame(data)

    # Create figure with specific size
    plt.figure(figsize=(10, 6))

    # Create the stacked bars
    ax = df.plot(
        x="Model",
        y=["Correct", "Ambiguous", "Incorrect"],
        kind="bar",
        stacked=True,
        color=["#2ecc71", "#95a5a6", "#e74c3c"],
        width=0.65,  # Make bars slightly thinner
    )

    # Customize the plot
    plt.title("Model Performance Comparison", pad=20, fontsize=14, fontweight="bold")
    plt.xlabel("Model", fontsize=12, labelpad=10)
    plt.ylabel("Percentage", fontsize=12, labelpad=10)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Set y-axis limits and grid
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add percentage labels in the middle of each segment
    for c in ax.containers:
        # Add labels with white color for better visibility
        labels = ax.bar_label(c, fmt="%.1f%%", label_type="center", color="white", fontweight="bold", fontsize=10)

    # Enhance legend
    plt.legend(
        title="Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        title_fontsize=12,
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show plot
    plt.show()


def calculate_accuracy(data: List[Dict], identifier: str) -> float:
    if type(data) == pd.DataFrame:
        data = data.to_dict(orient="records")
    total_comparisons = 0
    correct_predictions = 0
    equal_predictions = 0
    incorrect_predictions = 0

    # Track accuracy per row and per clause
    row_accuracies = {}  # uuid -> accuracy
    clause_accuracies = {}  # clause -> accuracy

    # Get all unique clause bases (without the A/B suffix)
    clauses = [x for x in data[0].keys() if x.startswith("Reward") or x.startswith("Penalize")]

    for record in data:
        row_correct = 0
        row_total = 0

        for clause in clauses:
            clause_correct = 0
            clause_total = 0

            a_key = f"{identifier}_A_{clause}"
            b_key = f"{identifier}_B_{clause}"

            if a_key in record and b_key in record:
                a_score = record[a_key]
                b_score = record[b_key]

                # Skip equal predictions for accuracy calculation
                if a_score == b_score:
                    equal_predictions += 1
                    continue

                total_comparisons += 1
                row_total += 1
                clause_total += 1

                if (a_score > b_score and record[clause] == "A") or (b_score > a_score and record[clause] == "B"):
                    correct_predictions += 1
                    row_correct += 1
                    clause_correct += 1
                else:
                    incorrect_predictions += 1

            # Calculate clause accuracy
            if clause_total > 0:
                accuracy = clause_correct / clause_total
                if clause not in clause_accuracies:
                    clause_accuracies[clause] = []
                clause_accuracies[clause].append(accuracy)

        # Calculate row accuracy
        if row_total > 0:
            row_accuracies[record["uuid"]] = row_correct / row_total

    total = correct_predictions + equal_predictions + incorrect_predictions

    # Calculate percentages rounded to 1 decimal place
    correct_pct = round((correct_predictions / total) * 100, 1) if total > 0 else 0
    equal_pct = round((equal_predictions / total) * 100, 1) if total > 0 else 0
    incorrect_pct = round(100 - correct_pct - equal_pct, 1) if total > 0 else 0

    # Find lowest performing row and clause
    if row_accuracies:
        worst_row_uuid = min(row_accuracies.items(), key=lambda x: x[1])[0]
        worst_row_accuracy = row_accuracies[worst_row_uuid]

    if clause_accuracies:
        # Calculate average accuracy for each clause
        clause_avg_accuracies = {clause: np.mean(accuracies) for clause, accuracies in clause_accuracies.items()}
        worst_clause = min(clause_avg_accuracies.items(), key=lambda x: x[1])[0]
        worst_clause_accuracy = clause_avg_accuracies[worst_clause]

    print(f"Results for {identifier}:")
    print(f"Correct: {correct_pct}%")
    print(f"Ambiguous: {equal_pct}%")
    print(f"Incorrect: {incorrect_pct}%")
    print("---")

    return correct_pct, equal_pct, incorrect_pct


def gpt_make_inference_request(clause, messages, response):
    from google.colab import userdata
    import re

    SYSTEM_MESSAGE = """
# Task
Your task is strictly to analyze and score the LLM asssistant's latest message, strictly with reference to the criteria clause provided. Do not engage with the conversation, your role is only to analyze and score.

Scores are on a continuous scale between 0 and 1, you should report the score to two decimal places.

# Criteria
{clause}

# Output format
Analyse how well the assistant's final message performs against the clause criteria, then provide a final score between 0 and 1 wrapped in XML tags as shown in the example below
<score>0.5</score>

# Scoring guidance
- A score of 1 indicates that the assistant's message is perfect with respect to the criteria clause. There is no room for improvement.
- A score between 0.75 and 0.99 indicates that the assistant's message is very good and mostly excels in regards to the criteria clause.
- A score between 0.5 and 0.74 indicates that the assistant's message is good and meets the criteria clause to a reasonable extent.
- A score between 0.25 and 0.49 indicates that the assistant's message is poor and does not meet the criteria clause to a reasonable extent.
- A score between 0 and 0.24 indicates that the assistant's message is very poor and fails to meet the criteria clause.
"""

    client = OpenAI(
        api_key=userdata.get("OPENAI_API_KEY"),
    )
    user_message = "\n\n".join(
        [
            f"<{msg.role}>\n{msg.content}\n</{msg.role}>"
            for msg in messages + [{"role": "assistant", "content": response}]
        ]
    )
    system_message = SYSTEM_MESSAGE.format(clause=clause.__str__())
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"```\n{user_message}\n```"},
        ],
        model="gpt-4o",
        temperature=0.0,
    )

    match = re.search(r"<score>([\d\.]+)</score>", chat_completion.choices[0].message.content)
    score = float(match.group(1))
    return score


