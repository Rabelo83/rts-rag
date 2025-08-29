import requests

URL = "http://127.0.0.1:8000/ask"
print("Type your question (or 'exit'):")
while True:
    q = input("> ").strip()
    if q.lower() in {"exit","quit"} or not q:
        break
    try:
        r = requests.post(URL, json={"question": q, "k": 5}, timeout=180)
        r.raise_for_status()
        data = r.json()
        print("\nAnswer:", data.get("answer",""))
        if data.get("sources"):
            print("Sources:")
            for s in data["sources"]:
                print(" -", s)
        print()
    except Exception as e:
        print("Error:", e)
