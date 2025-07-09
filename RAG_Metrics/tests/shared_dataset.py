# tests/shared_dataset.py
import sys
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from RAG_Metrics.rag_cli import retrieve
from datasets import Dataset

_rows = [
    {
        "question": "What is the Galaxy S22’s water- and dust-resistance rating?",
        "reference": "IP68",
    },
    {
        "question": "Which models accept a 45 W charger for Super-Fast Charging?",
        "reference": "Galaxy S22+ and Galaxy S22 Ultra",
    },
    {
        "question": "How can you charge another device with your phone?",
        "reference": "Wireless power sharing",
    },
    {
        "question": "Which key combination takes a screenshot?",
        "reference": "Press and release the Side and Volume down keys",
    },
    {
        "question": "What shooting-mode lets the camera pick the best photo settings?",
        "reference": "Photo mode",
    },
    {
        "question": "What feature boosts microphone gain while you zoom video?",
        "reference": "Zoom-in mic",
    },
    {
        "question": "What do ‘Air actions’ let the S Pen do?",
        "reference": "Perform remote functions with the S Pen button and gestures",
    },
]

# Run your retrieval once to attach contexts
for row in _rows:
    row["contexts"] = [hit["text"] for hit in retrieve(row["question"])]

dataset = Dataset.from_list(
    [
        {
            "question": r["question"],
            "answer": "",               # let the LLM fill this in later, or keep blank
            "reference": r["reference"],
            "contexts": r["contexts"],
        }
        for r in _rows
    ]
)
