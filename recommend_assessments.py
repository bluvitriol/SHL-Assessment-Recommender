import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
import requests
from bs4 import BeautifulSoup
import pprint
import pickle
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyAB_HZKai0J8mBgLDSSrp3KJ5Qn4A_2OrQ"))
model = genai.GenerativeModel("gemini-pro")



# Load model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("shl_faiss_index.index")

# Load assessment metadata
with open("shl_metadata.pkl", "rb") as f:
    documents = pickle.load(f)

def extract_link_from_query(text):
    url_pattern = r"(https?://[\w./\-=?#&%]+)"
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def extract_job_description_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        target_headings = [
            "What You Will Be Doing",
            "What we are looking for from you",
            "Desirable"
        ]

        extracted_description = ""
        for strong_tag in soup.find_all("strong"):
            heading_text = strong_tag.get_text(strip=True)
            if any(heading_text.lower() in th.lower() for th in target_headings):
                ul_tag = strong_tag.find_next("ul")
                if ul_tag:
                    for li in ul_tag.find_all("li"):
                        text = li.get_text(strip=True)
                        extracted_description += f"- {text}"

        return extracted_description.strip() if extracted_description else None
    except Exception as e:
        print(f"Coudln't extract job description from {url} \n {e}")
        return None

def extract_filters(query):
    filters = {
        "max_duration": None,
        "test_types": [],
        "keywords": [],
        "adaptive_required": None,
        "remote_required": None,
    }

    duration_match = re.search(r"(?:within|under|less than|up to|maximum|max)?\s*(\d{1,3})\s*(minutes|min)", query.lower())
    if duration_match:
        filters["max_duration"] = int(duration_match.group(1))

    keyword_map = {
        "java": "Knowledge & Skills",
        "python": "Knowledge & Skills",
        "sql": "Knowledge & Skills",
        "javascript": "Knowledge & Skills",
        "cognitive": "Ability & Aptitude",
        "aptitude": "Ability & Aptitude",
        "situational": "Biodata & Situational Judgement",
        "personality": "Personality & Behavior",
        "analyst": "Ability & Aptitude",
        "communication": "Personality & Behavior",
        "collaborate": "Personality & Behavior",
        "development": "Development & 360",
    }

    for word, test_type in keyword_map.items():
        if word in query.lower():
            filters["test_types"].append(test_type)
            filters["keywords"].append(word)

    if "adaptive" in query.lower() or "irt" in query.lower():
        filters["adaptive_required"] = "Yes"

    if "remote" in query.lower():
        filters["remote_required"] = "Yes"

    return filters

def recommend_assessments(user_input, top_k=20):
    
    link = extract_link_from_query(user_input)
    if link:
        jd_text = extract_job_description_from_url(link)
        if jd_text:
            user_input += "\n" + jd_text

    filters = extract_filters(user_input)
    

    prompt = f"Based on the following job description, identify the key skills and suggest appropriate assessment types:\n\n{user_input}"
    try:
        response = model.generate_content(prompt)
        enhanced_input = response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        enhanced_input = user_input 


    query_embedding = model.encode([enhanced_input])
    distances, indices = index.search(query_embedding, top_k)
    recommended = []
    count=0

    for idx in indices[0]:
        assessment = documents[idx]

        if filters["max_duration"] is not None and isinstance(assessment["duration"], int):
            if assessment["duration"] > filters["max_duration"]:
                continue

        if filters["test_types"]:
            if not any(tt in assessment["test_type"] for tt in filters["test_types"]):
                continue

        if filters["remote_required"] and assessment["remote_support"] != filters["remote_required"]:
            continue

        if filters["adaptive_required"] and assessment["adaptive_support"] != filters["adaptive_required"]:
            continue

        count+=1

        recommended.append({
            "url": assessment["url"],
            "adaptive_support": assessment["adaptive_support"],
            "description": assessment["description"],
            "duration": assessment["duration"],
            "remote_support": assessment["remote_support"],
            "test_type": assessment["test_type"]
        })

        if len(recommended) == 10:
            break


    if count==0:
        print("No relevant assessment found!")
    return recommended

#testing
if __name__ == "__main__":
    # pprint.pprint(recommend_assessments("Hiring for analyst role with cognitive and personality tests under 30 mins"))
    pprint.pprint(recommend_assessments("Here is a JD text: https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes."))

