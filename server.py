from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, re, tempfile, json, requests
from pathlib import Path
from PyPDF2 import PdfReader
import docx, git
from rapidfuzz import fuzz
import google.generativeai as genai

# Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)  # allow React frontend to access backend
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB max

ALLOWED_EXTENSIONS = {'.pdf', '.docx'}

def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def extract_text_from_file(file):
    text = ""
    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        if ext == ".pdf":
            reader = PdfReader(tmp.name)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:  # docx
            doc = docx.Document(tmp.name)
            text = "\n".join([p.text for p in doc.paragraphs])
    return text

def extract_github_links(text):
    return re.findall(r'(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', text)

def normalize_code(code):
    return re.sub(r"\s+", " ", code.strip())

def github_code_search(snippet, language=None, per_page=3):
    q = f'"{snippet}"'
    if language:
        q += f' language:{language}'
    url = "https://api.github.com/search/code"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    r = requests.get(url, params={"q": q, "per_page": per_page}, headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json().get("items", [])
    return []

def call_gemini_judge(candidate_snippet, top_matches_info):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a code originality analyst.

Candidate code snippet:
\"\"\"{candidate_snippet[:2000]}\"\"\"

Top matches:
{json.dumps(top_matches_info, indent=2)}

Return JSON with:
{{ "verdict":"Copied"|"Possibly Copied"|"No Match", "confidence":0.0-1.0, "top_evidence":[{{"url","similarity"}}], "reasoning":"short" }}
"""
    resp = model.generate_content(prompt)
    try:
        return json.loads(resp.text)
    except:
        return {"verdict":"Unknown","confidence":0.0,"top_evidence":[],"reasoning":resp.text}

@app.route("/analyze", methods=["POST"])
def analyze():
    jd = request.form.get("job_description", "")
    uploaded_files = request.files.getlist("resumes")
    results = []

    for file in uploaded_files:
        if not allowed_file(file.filename):
            results.append({"filename": file.filename, "error": "File type not allowed"})
            continue

        resume_text = extract_text_from_file(file)
        github_links = extract_github_links(resume_text)

        # Evaluate resume vs Job Description
        prompt = f"""
Job Description:
{jd}

Resume:
{resume_text}

Provide:
1) Suitability score out of 10
2) Key strengths
3) Missing skills/weaknesses
"""
        resume_eval = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text

        # Analyze GitHub repos
        repo_results = []
        for repo_url in github_links:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    repo_dir = Path(tmpdir)/"repo"
                    git.Repo.clone_from(repo_url, repo_dir, depth=1)
                    for fpath in Path(repo_dir).rglob("*"):
                        if fpath.suffix.lower() in {".py",".js",".java",".cpp",".ts"}:
                            code = fpath.read_text(errors="ignore")
                            snippet = "\n".join([l for l in code.splitlines() if l.strip()][:10])
                            snippet_norm = normalize_code(snippet)
                            lang = fpath.suffix.lstrip(".")
                            matches = github_code_search(snippet_norm, language=lang)
                            top_matches = [{"url": m.get("html_url"), "similarity": fuzz.token_set_ratio(snippet_norm, snippet_norm)/100.0} for m in matches]
                            if top_matches:
                                verdict = call_gemini_judge(snippet_norm, top_matches)
                            else:
                                verdict = {"verdict":"No Match","confidence":1.0,"top_evidence":[],"reasoning":"No similar code found"}
                            repo_results.append({
                                "file": str(fpath.relative_to(repo_dir)),
                                "verdict": verdict["verdict"],
                                "confidence": verdict["confidence"],
                                "top_evidence": verdict.get("top_evidence", []),
                                "reasoning": verdict.get("reasoning", "")
                            })
            except Exception as e:
                repo_results.append({"repo_url": repo_url, "error": str(e)})

        results.append({
            "filename": file.filename,
            "resume_analysis": resume_eval,
            "github_analysis": repo_results,
            "github_links": github_links
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
