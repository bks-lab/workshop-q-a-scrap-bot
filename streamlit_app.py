import streamlit as st
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import sys
import os
from dotenv import load_dotenv

# Laden der Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Konfiguration der OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"

# Einlesen der Daten aus der CSV-Datei
path_csv_Data = './embeddings.csv'
df = pd.read_csv(path_csv_Data, on_bad_lines='skip')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# Textregeln
regel1 = '''Beantworte die Frage so wahrheitsgemäß wie möglich nur mit den Informationen, die im KONTEXT stehen. 
Strukturiere und formatiere die Antwort leserlich und visuell mit den Informationen, die zur Verfügung stehen.
Ändere den Satzaufbau, um den Fragesteller besser zu informieren.
LINKS und EMAILs, die aus dem KONTEXT benutzt werden dürfen, dürfen NICHT VERÄNDERT werden.'''

regel2 = '''Wenn die Antwort nicht im KONTEXT zu finden ist, beantworte die Frage nicht und sage: 
"Tut mir leid, leider konnte ich zu Ihrer Frage keine Daten finden."'''

prepromt = regel1 + regel2

# Funktion zur Generierung der Antwort
def get_response(message, df, model):
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, get_embedding(message, engine="text-embedding-ada-002")))
    top_3_answers = df.sort_values("similarities", ascending=False).iloc[:3]['Sentence'].tolist()
    
    sim_kontext = 'KONTEXT:\n'    
    for i, answer in enumerate(top_3_answers):
        sim_kontext += answer + '\n'
        
    messages = [
        {"role": "system", "content": "Du bist ein hilfreicher Website Q&A Bot." + prepromt},
        {"role": "user", "content": message + '\n' + sim_kontext}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    response_text = response["choices"][0]["message"]["content"]

    return response_text

# Streamlit App
def main():
    st.title("Frage-Antwort-Bot")
    st.write("Stellen Sie eine Frage und erhalten Sie eine Antwort!")
    
    question = st.text_input("Frage eingeben:")
    
    if st.button("Antwort erhalten"):
        if question:
            answer = get_response(question, df, model)
            st.write("Antwort:")
            st.write(answer)
        else:
            st.write("Bitte geben Sie eine Frage ein.")

if __name__ == "__main__":
    main()
