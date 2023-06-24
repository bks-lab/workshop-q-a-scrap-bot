# workshop-q-a-scrap-bot
Hier der Code aus der KI-Presentation: https://pitch.com/public/bc7eaebb-5fb1-4e6f-bf84-ce9363c7abc0

Folgender Code ist hier zu finden

## Python-Code zur Textembedding-Extraktion und -Speicherung

Dieser Python-Code demonstriert eine Funktionalität zum Extrahieren von Sätzen aus einer Webseite und dem anschließenden Einbetten der Sätze mithilfe des OpenAI-Modells "embedding-adda-002". Die extrahierten Sätze und ihre Embeddings werden dann in einer CSV-Datei gespeichert.

### Funktionalität:

1. Der Code extrahiert Sätze aus einer zuvor generierten CSV-Datei (`sentences.csv`). Diese CSV-Datei sollte Sätze enthalten, die zuvor aus Webseiten extrahiert wurden.
2. Die Funktion `embed_text(text)` verwendet den OpenAI API-Schlüssel, um den gegebenen Text mit dem Modell "embedding-adda-002" einzubetten und gibt das Embedding zurück.
3. Die Funktion `embed_csv(input_file, output_file)` liest die Sätze aus der Eingabe-CSV-Datei ein und embeddet sie einzeln mithilfe der `embed_text()`-Funktion. Die Sätze und ihre Embeddings werden dann in einer Ausgabe-CSV-Datei (`embedded_sentences.csv`) gespeichert.
4. Der Code kann durch Ausführen der `main()`-Funktion gestartet werden. Dabei werden die zuvor generierte CSV-Datei (`sentences.csv`) eingelesen, die Embeddings berechnet und in der Ausgabe-CSV-Datei (`embedded_sentences.csv`) gespeichert.

### Anwendung:

1. Erstelle eine CSV-Datei (`sentences.csv`) mit zuvor extrahierten Sätzen aus Webseiten.
2. Füge deinen OpenAI API-Schlüssel in den Code ein.
3. Führe den Code aus, um die Sätze einzubetten und die Embeddings in einer Ausgabe-CSV-Datei (`embedded_sentences.csv`) zu speichern.

Bitte beachte, dass du die OpenAI Python-Bibliothek installieren und deinen API-Schlüssel in den Code einfügen musst, um die OpenAI-Funktionalität nutzen zu können.

## Verctorisierung der Daten

Die Funktionalität in diesem Codeausschnitt besteht aus den folgenden Schritten:

1. Importiere die Funktion `get_embedding` aus dem Modul `openai.embeddings_utils`.
2. Setze den Index des DataFrames `df` zurück und aktualisiere den DataFrame, sodass der Index neu nummeriert wird.
3. Füge eine neue Spalte namens 'embedding' zum DataFrame `df` hinzu.
4. Verwende die Methode `apply` auf die Spalte 'Sentence' des DataFrames, um die Funktion `get_embedding` auf jeden Wert der Spalte anzuwenden.
5. Die Funktion `get_embedding` nimmt den Textwert `x` und den optionalen Parameter `engine` entgegen und gibt das entsprechende Embedding zurück.
6. Das Embedding wird mit dem Wert 'text-embedding-ada-002' generiert, der als Engine angegeben ist.
7. Das generierte Embedding wird der 'embedding'-Spalte des DataFrames `df` zugewiesen.
8. Speichere den aktualisierten DataFrame `df` als CSV-Datei mit dem Namen 'embeddings.csv' im aktuellen Verzeichnis.
9. Gib den aktualisierten DataFrame `df` aus.


## Funktionalität des Q&A-Bots

Dieser Q&A-Bot basiert auf dem GPT-3.5-Turbo-Modell von OpenAI und verwendet eine vorgefertigte Liste von Embeddings, um Fragen zu beantworten. Hier ist eine Übersicht über den Code und seine Funktionalität:

1. Importieren der benötigten Bibliotheken und Module: `openai`, `pandas`, `numpy`, `sys`, `os` und `dotenv`.

2. Laden der Umgebungsvariablen aus der `.env`-Datei mithilfe von `load_dotenv()`.

3. Lesen der Embedding-Daten aus der CSV-Datei und Speichern in einem Pandas DataFrame (`df`).

4. Setzen des OpenAI-API-Schlüssels aus der Umgebungsvariable `OPENAI_API_KEY`.

5. Definieren von Regeln (`regel1` und `regel2`) und einer Prompt-Vorlage (`prepromt`), die dem Bot helfen, die Antworten strukturiert und formatiert zurückzugeben.

6. Erstellen der Funktion `get_response(message, df, model)`, die die eigentliche Verarbeitung der Frage und Generierung der Antwort durchführt:

    - Berechnen der Ähnlichkeiten zwischen der Frage (`message`) und den Embeddings im DataFrame mithilfe der `cosine_similarity`-Funktion.

    - Extrahieren der drei besten Übereinstimmungen (`top_3_answers`) aus dem DataFrame basierend auf den Ähnlichkeiten.

    - Erstellen eines KONTEXT-Textes (`sim_kontext`), der die drei besten Antworten enthält.

    - Anzeigen der top 3 Antworten und des KONTEXT-Textes als Loggings.

    - Erstellen von Systemnachrichten (`role: system`) und Benutzernachrichten (`role: user`) für die Kommunikation mit dem GPT-3.5-Turbo-Modell.

    - Verwenden der `openai.ChatCompletion.create()`-Methode, um eine Antwort vom Modell zu erhalten.

    - Extrahieren des Antworttexts aus der API-Antwort.

    - Rückgabe der generierten Antwort.

Bitte beachten Sie, dass der Bot die Antwort basierend auf den Informationen im KONTEXT formuliert. Wenn die Antwort nicht im KONTEXT zu finden ist, wird eine Standardmeldung zurückgegeben.

Wenn Sie den Code verwenden möchten, stellen Sie sicher, dass Sie den OpenAI-API-Schlüssel in der `.env`-Datei hinterlegt haben und die erforderlichen Bibliotheken installiert sind.

## Frage-Antwort-Bot-Streamlit

Diese Streamlit-Anwendung ermöglicht es Ihnen, eine Frage einzugeben und eine Antwort zu erhalten. Der Bot verwendet die OpenAI GPT-3.5-Turbo-Modell, um die Frage zu analysieren und basierend auf dem vorhandenen Kontext die beste Antwort zu generieren.

### Anwendung

1. Geben Sie Ihre Frage in das Eingabefeld ein.
2. Klicken Sie auf "Antwort erhalten", um die Frage an den Bot zu senden.
3. Der Bot analysiert Ihre Frage und den vorhandenen Kontext, um eine passende Antwort zu generieren.
4. Die generierte Antwort wird unterhalb des Eingabefelds angezeigt.

### Hinweise

- Stellen Sie sicher, dass Ihre Frage präzise und verständlich formuliert ist, um genauere Antworten zu erhalten.
- Der Bot verwendet den vorhandenen Kontext, um die Antworten zu generieren. Die Qualität der Antworten hängt daher von der Qualität und Relevanz des Kontexts ab.
- Wenn die Antwort nicht im Kontext zu finden ist, wird der Bot keine Daten liefern und eine entsprechende Nachricht anzeigen.






