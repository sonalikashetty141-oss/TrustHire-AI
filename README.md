# TrustHire AI 🛡️

**TrustHire AI** is an educational portfolio project that demonstrates how Python and machine learning can help users review job and internship postings for potential warning signs.

> **Important limitation:** This prototype is not a fraud detector, does not verify employers, and cannot determine whether a listing is safe. Its training dataset is a tiny hand-written sample for learning only. Do not make real-world decisions based on its score.

## Features

- Streamlit web interface for pasting a job description
- TF-IDF text features and Logistic Regression classifier
- Separate checklist for common warning phrases
- Basic safety checklist for applicants
- Small sample dataset included so the project runs without a dataset download

## Project structure

```text
TrustHire_AI/
├── app.py
├── requirements.txt
├── README.md
├── test_results.md
├── .gitignore
├── data/
│   └── sample_jobs.csv
└── tests/
    └── test_app_helpers.py
```

## 1. Install Python

Use Python 3.11 if possible. Check the version:

```bash
python --version
```

## 2. Open the project in VS Code

Extract the ZIP, open the `TrustHire_AI` folder in VS Code, then open the terminal inside that folder.

## 3. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can avoid changing execution policy and run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Install dependencies

With the environment activated:

```bash
python -m pip install -r requirements.txt
```

## 5. Run the app

```bash
python -m streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open that address in your browser while the terminal process is still running.

## 6. Try a safe test example

Paste this text:

> Junior Python intern. The role includes mentoring, a technical interview, and an application through the company's official careers portal. No application fee is charged.

Then try a suspicious example:

> Guaranteed job with huge income. Pay a registration fee today and send your OTP to activate your salary.

The demo should flag the second text's warning phrases. The model's score is only illustrative because the dataset is extremely small.

## 7. Run tests

Install pytest if needed:

```bash
python -m pip install pytest
python -m pytest -q
```

## 8. Improve the dataset responsibly

The included CSV has two columns:
- `text`: the job-posting text
- `label`: `0` for the example labelled legitimate and `1` for the example labelled fraudulent

These labels are only illustrative. For a credible ML experiment:
1. Obtain a public, appropriately licensed labelled job-posting dataset from a reputable source.
2. Read its license, data dictionary, and intended use.
3. Remove personal data and irrelevant fields.
4. Use a stratified train/test split.
5. Evaluate precision, recall, F1, and a confusion matrix.
6. Check for duplicates and data leakage.
7. Never describe the score as a verified probability unless it has been calibrated and validated.

Do not report performance metrics from the tiny included sample as evidence of real-world accuracy.

## 9. Deploy

A simple option is Streamlit Community Cloud:
1. Create a GitHub repository and upload this project.
2. Confirm `app.py`, `requirements.txt`, `README.md`, and `data/sample_jobs.csv` are in the repository.
3. Sign in to Streamlit Community Cloud.
4. Create a new app, select the repository and `app.py`, then deploy.
5. Test the public URL in a private/incognito browser window.

The first build may take a few minutes. Keep the repository public only if you're comfortable sharing all its files. Never upload passwords, API keys, or private data.

## 10. Suggested LinkedIn post

> 🚀 Introducing TrustHire AI — a Python + Machine Learning portfolio project.
>
> I built a Streamlit prototype that reviews job and internship descriptions for potential warning signs using TF-IDF, Logistic Regression, and a transparent rules-based checklist.
>
> This project helped me practice text classification, building a user interface, testing, and responsible AI communication. It is an educational prototype, not a guarantee that a job is genuine or fraudulent.
>
> #Python #MachineLearning #Streamlit #ArtificialIntelligence #BuildInPublic

Add your GitHub repository and live demo links after you have uploaded and tested them.

## Future improvements

- Replace sample rows with a licensed, real-world dataset
- Add proper train/test evaluation and a confusion matrix
- Add a source-verification checklist without automatically labelling companies
- Improve accessibility and mobile layout
- Add a privacy notice and avoid storing pasted descriptions
