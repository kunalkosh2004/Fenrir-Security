import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import os

def generate_cli_qa_dataset(url) -> List[Dict]:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  

    soup = BeautifulSoup(response.content, "html.parser")

    qa_pairs = []

    for h3 in soup.find_all("h3"):
        question = h3.get_text(strip=True)

        answer_parts = []
        for sibling in h3.find_next_siblings():
            if sibling.name == "h3":
                break
            if sibling.name == "p":
                answer_parts.append(sibling.get_text(strip=True))
        answer = " ".join(answer_parts)

        if question and answer:
            qa_pairs.append({
                "question": question,
                "answer": answer
            })

    return qa_pairs

def save_dataset(qa_pairs: List[Dict], file_path: str):
    """Save Q&A pairs to JSON file"""
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {file_path}")

def main():
    url1 = "https://www.wecreateproblems.com/interview-questions/linux-commands-interview-questions"
    url2 = "https://www.testgorilla.com/blog/linux-commands-interview-questions/"
    url3 = "https://www.geeksforgeeks.org/git-interview-questions-and-answers/"
    url4 = "https://zerotomastery.io/blog/bash-scripting-interview-questions/"
    url5 = "https://builtin.com/articles/grep-command"
    url6 = "https://www.simplilearn.com/linux-commands-interview-questions-article"
    qa1 = generate_cli_qa_dataset(url1)
    qa2 = generate_cli_qa_dataset(url2)
    qa3 = generate_cli_qa_dataset(url3)
    qa4 = generate_cli_qa_dataset(url4)
    qa5 = generate_cli_qa_dataset(url5)
    qa6 = generate_cli_qa_dataset(url6)
    qa_final = qa1 + qa2 + qa3 + qa4 + qa5 + qa6
    
    save_dataset(qa_final,"./data/cli_qa_dataset.json")


if __name__ == "__main__":
    main()

