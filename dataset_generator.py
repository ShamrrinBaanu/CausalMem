"""Create a small synthetic benchmark fixture; never loaded by the live app."""
import json
import argparse
from pathlib import Path

CASES = [
    {"case_id":"temporal-python-java","conversation":["In 2024 I started learning Python.","In 2025 I began learning Java after Python."],"question":"What did I learn before Java?","category":"TEMPORAL","expected_any":["python"]},
    {"case_id":"explicit-cause-recursion","conversation":["I struggled with recursion in Python.","I started solving coding problems because I wanted to improve at recursion."],"question":"Why did I start solving coding problems?","category":"CAUSAL","expected_any":["recursion","improve"]},
    {"case_id":"preference-change","conversation":["I prefer Python for personal projects.","I now prefer Java for backend projects."],"question":"What language do I currently prefer for backend projects?","category":"PREFERENCE","expected_any":["java"]},
    {"case_id":"activity-history","conversation":["I began learning Python.","I started solving LeetCode problems."],"question":"What activity did I start after learning Python?","category":"HISTORY","expected_any":["leetcode"]},
    {"case_id":"interest-evolution","conversation":["I was interested in Python scripting.","Recently I became interested in Java backend development."],"question":"How did my programming interests evolve?","category":"EVOLUTION","expected_any":["python","java"]},
]

def generate(path="data/evaluation_questions.json"):
    """Create a fixture at path without replacing any existing evaluation data."""
    target=Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as stream:
        json.dump(CASES,stream,indent=2)
    return target

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Write the synthetic benchmark fixture without overwriting existing data.")
    parser.add_argument("output",nargs="?",default="data/evaluation_questions.json",help="New dataset path")
    args=parser.parse_args()
    print(generate(args.output))
