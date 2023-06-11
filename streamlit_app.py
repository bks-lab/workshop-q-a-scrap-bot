import streamlit as st
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import sys
import os
from dotenv import load_dotenv

path_csv_Data = './embeddings.csv'
df = pd.read_csv(path_csv_Data, on_bad_lines='skip')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"

regel1 = ''' Beantworte die Frage so wahrheitsgemäß wie möglich nur mit den Informationen die im KONTEXT stehen. 
    Strukturiere und Formatiere die Antwort leserlich und visuell mit den Informatione die zur verfügung stehen.
    Und ändere den Satzaufbau um den Fragesteller besser zu informieren.
    LINKS und EMAILs die aus dem KONTEXT benutzt werden dürfen NICHT VERÄNDERT werden. '''

regel2 = '\n Wenn die Antwort nicht im KONTEXT zu finden ist so beantworte die Frage nicht und sage: "Tut mir ' \
    'leid, leider konnte ich zu Ihrer Frage keine Daten finden."'
prepromt = regel1 + regel2


def get_response(message, df, model):
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, get_embedding(message, engine="text-embedding-ada-002")))
    top_3_answers = df.sort_values("similarities", ascending=False).iloc[:3]['Sentence'].tolist()
    
    sim_kontext = 'KONTEXT:\n'    
    for i, answer in enumerate(top_3_answers):
        sim_kontext += f"{i + 1}: {answer}\n"
    
    messages = [
        {"role": "system", "content": "Du bist ein hilfreicher Website Q&A Bot." + prepromt},
        {"role": "user", "content": message + '\n' + sim_kontext}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5
    )

    response_text = response["choices"][0]["message"]["content"]

    return response_text


def main():
    st.title("Frage-Antwort-Anwendung")

    st.sidebar.markdown("## Erklärung")
    st.sidebar.write("Diese Anwendung ermöglicht es, Fragen zu stellen und Antworten zu erhalten.")
    st.sidebar.write("Die Antworten werden basierend auf dem gegebenen Kontext generiert.")
    st.sidebar.write("Bitte stellen Sie die Frage im Textfeld unten.")

    question = st.text_input("Frage eingeben", "")

    if st.button("Antwort erhalten"):
        response = get_response(question, df, model)
        st.write("Antwort:")
        st.write(response)


if __name__ == '__main__':
    main()
