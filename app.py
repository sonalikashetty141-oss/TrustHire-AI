import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "trusthire_model.joblib"
DATA_PATH = ROOT / "data" / "sample_jobs.csv"

st.set_page_config(page_title="TrustHire AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }

    h1 {
        color: #c62828;
        font-weight: 800;
    }

    h2, h3 {
        color: #b71c1c;
    }

    div[data-testid="stMetric"] {
        background-color: #fff5f5;
        border: 1px solid #ffcdd2;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button[kind="primaryFormSubmit"] {
        background-color: #c62828;
        color: white;
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ TrustHire AI")
st.caption("An AI-powered assistant to help you identify potential job scams and evaluate job postings more carefully.")
st.markdown("""
<div style="
    background: #fff5f5;
    border-left: 5px solid #c62828;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 20px;
">
    <h3 style="margin-top: 0; color: #b71c1c;">
        Your Smart Job Safety Assistant
    </h3>
    <p style="margin-bottom: 0; color: #333333;">
        Analyze job and internship postings, identify potential warning signs,
        and learn practical ways to verify employers before applying.
    </p>
</div>
""", unsafe_allow_html=True)
st.subheader("Make more informed decisions before applying for a job")
st.info(
    "Educational prototype only. A score is not proof that a job is fraudulent or safe. "
    "Always verify employers independently and never pay to obtain a job."
)

WARNING_PATTERNS = {
    "Upfront payment or fee": r"\b(upfront fee|registration fee|pay to apply|training fee|processing fee|security deposit|pay a fee before joining)\b",

    "Unusually guaranteed income": r"\b(guaranteed income|guaranteed selection|earn\s+₹?\$?\d[\d,]*(?:\.\d+)?\s*(?:per day|daily|per week|weekly)|get rich quick|unusually high pay)\b",

    "Sensitive information request": r"\b(send your password|bank password|one[- ]time password|otp|full card number)\b",

    "Pressure or urgency": r"\b(apply immediately|limited time only|urgent hiring|act now)\b",

    "Vague employer details": r"\b(no previous experience is needed|no interview required)\b",

    
    "Upfront payment or fee": r"\b(upfront fee|registration fee|pay to apply|training fee|processing fee|security deposit|pay a fee before joining|refundable deposit|refundable registration deposit|pay before receiving job details|payment required before joining)\b",

    "Unusually guaranteed income": r"\b(guaranteed income|guaranteed selection|earn\s+₹?\$?\d[\d,]*(?:\.\d+)?\s*(?:per day|daily|per week|weekly)|get rich quick|unusually high pay|earn money easily|high daily salary)\b",

    "Sensitive information request": r"\b(send your password|bank password|one[- ]time password|otp|full card number|bank account details|share your banking details|send your personal financial information)\b",

    "Pressure or urgency": r"\b(apply immediately|limited time only|urgent hiring|act now|apply right now|last chance to apply|secure your position immediately)\b",

    "Vague employer details": r"\b(no previous experience is needed|no interview required|guaranteed job without interview|job details will be provided after payment)\b",

    "Unofficial communication channel": r"\b(through whatsapp|via whatsapp|whatsapp me|contact only on whatsapp|message this personal number|unofficial messaging account)\b",
}


@st.cache_resource
def load_or_train_model():
    data = pd.read_csv(DATA_PATH)

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain text and label columns.")

    if data["text"].isna().any() or data["label"].isna().any():
        raise ValueError("Dataset contains missing values.")

    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both labels: 0 and 1.")

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    model.fit(data["text"], data["label"])
    joblib.dump(model, MODEL_PATH)
    return model

def find_warnings(text):
    found = []
    for label, pattern in WARNING_PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            found.append(label)
    return found

model = load_or_train_model()
if "analysis_submitted" not in st.session_state:
    st.session_state.analysis_submitted = False
st.divider()

st.markdown("## 📝 Enter Job Details")   

st.caption("💡 Add the job title, company name, and full description for a more useful review.")

with st.form("job_form"):
    title = st.text_input(
        "💼 Job Title (Optional)",
        placeholder="e.g., Junior Python Intern"
    )

    company = st.text_input(
        "🏢 Company Name (Optional)",
        placeholder="Enter the employer's name"
    )

    description = st.text_area(
        "Paste the job or internship description",
        height=180
    )

    submitted = st.form_submit_button(
    "🔍 Analyze Job Posting",
    type="primary",
    use_container_width=True
)
if submitted or st.session_state.analysis_submitted:
    if len(description.strip()) < 30:
        st.warning("Please paste a longer description (at least 30 characters) for a more useful demo.")
    else:
        st.session_state.analysis_submitted = True
        combined = f"Job title: {title}\nCompany: {company}\n{description}"
        probability = float(model.predict_proba([combined])[0][1])
        warnings = find_warnings(combined)

        st.divider()
        st.markdown(
    "<h3 style='color:#e63946;'>📊 Analysis</h3>",
    unsafe_allow_html=True
)

        col1, col2 = st.columns(2)
        col1.metric(
    "🚨 Model-estimated risk",
    f"{probability:.0%}",
    help="This is an experimental model estimate, not a confirmed scam probability."
)

        col2.metric(
    "⚠️ Warning-sign categories",
    len(warnings),
    help="Number of warning categories detected by the checklist."
)
        st.caption(
    "Note: The risk score is an experimental model estimate, "
    "not a confirmed probability that a job is a scam."
)
        st.markdown(
    "<h3 style='color:#e63946;'>🔍 Why was this job flagged?</h3>",
    unsafe_allow_html=True
)

        if warnings:
            st.warning(
    "⚠️ Potential warning signs detected. Please review them carefully "
    "before proceeding with this job application."
)

            for item in warnings:
                st.write(f"- **{item}**")

            st.info(
                "These warning signs deserve attention. "
                "Verify the employer and job offer independently before sharing "
                "personal information or making payments."
            )
        else:
            st.info(
        "No warning phrases from this checklist were detected. "
        "However, this does not guarantee that the job is legitimate. "
        "Please verify the employer, application process, and job details "
        "independently before sharing personal information or making payments."
    )
            
        st.divider()
        st.markdown(
    "<h3 style='color:#e63946;'>✅ Job Safety Checklist</h3>",
    unsafe_allow_html=True
)
        checklist_progress = st.progress(0)
        st.write("Review these points before applying:")

        checklist_items = [
            ("I verified the company through its official website.", "check_company"),
            ("I checked whether the recruiter uses an official company email.", "check_email"),
            ("I was not asked to pay a recruitment or registration fee.", "check_fee"),
            ("I was not asked to share passwords, OTPs, or bank details.", "check_private_info"),
            ("The job responsibilities and application process are clear.", "check_job_details"),
        ]

        for label, key in checklist_items:
            st.checkbox(label, key=key)

        completed_checks = sum(
            st.session_state.get(key, False)
            for _, key in checklist_items
        )

        checklist_progress.progress(completed_checks / len(checklist_items))

        st.write(f"Completed: {completed_checks} out of {len(checklist_items)} checks")
        
        
        st.caption(
    "Note: This is a demonstration model trained on sample data. "
    "Its risk score is an estimate, not a verified probability. "
    "Always verify job offers independently before making decisions."
)
        with st.expander("Safer verification checklist"):
            st.write("- Check that the job appears on the company's official careers website.")
            st.write("- Verify the recruiter using contact details from the official website.")
            st.write("- Be cautious if asked to pay fees, buy equipment from a specific seller, or share OTPs/passwords.")
            st.write("- Do not upload sensitive personal information to this demo.")
            st.write("- If unsure, ask a trusted adult, college placement office, or official support channel.")

with st.expander("How this prototype works"):
    st.write("1. TF-IDF converts words and phrases into numeric features.")
    st.write("2. Logistic Regression learns patterns from labelled sample descriptions.")
    st.write("3. A separate rules-based checklist highlights specific warning phrases.")
    st.write("The sample data is intentionally tiny for learning. Replace it with a properly sourced, labelled dataset before reporting meaningful metrics.")
