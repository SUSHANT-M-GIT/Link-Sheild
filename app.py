
import html as html_lib
import re
import pickle
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.feature_extraction import extract_features


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LinkShield | URL Checker",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "cnn_bilstm_fusion.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
SCALER_PATH = BASE_DIR / "models" / "feature_scaler.joblib"

MAX_LEN = 200

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_slashes",
    "num_params",
    "num_subdomains",
    "has_https",
    "has_ip",
    "has_at",
]

EXAMPLES = {
    "Wikipedia": "https://www.wikipedia.org",
    "Suspicious login": (
        "http://secure-login.account-verify.example-bank.com."
        "update-now.xyz/signin?id=88213"
    ),
    "Raw IP address": "http://192.168.10.45/paypal/login.php",
}

# Display thresholds, not additional model predictions
SAFE_BELOW = 0.35
DANGER_FROM = 0.65

RING_CIRCUMFERENCE = 339.292


# ============================================================
# 2. CUSTOM CSS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* IMPORTANT: Do not apply the font to every Streamlit class.
   Streamlit icons need their own Material Symbols font. */

html, body {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

div[data-testid="stMarkdownContainer"],
div[data-testid="stTextInput"],
div[data-testid="stForm"],
div[data-testid="stButton"],
div[data-testid="stFormSubmitButton"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

/* Preserve Streamlit Material icons */
span[data-testid="stIconMaterial"],
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    direction: ltr;
    font-feature-settings: 'liga';
    -webkit-font-feature-settings: 'liga';
}

/* Hide unnecessary Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Header */
.ls-brand {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 8vw, 3.6rem);
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0;
}

.ls-brand span {
    color: #0e9aa5;
}

.ls-tagline {
    font-size: 1.05rem;
    opacity: 0.72;
    margin: 0.7rem 0 1.8rem 0;
}

/* Input form */
div[data-testid="stForm"] {
    border: 1px solid rgba(127,127,127,0.28);
    border-radius: 14px;
    padding: 1.1rem;
    background: rgba(127,127,127,0.06);
}

div[data-testid="stTextInput"] input {
    font-size: 1rem;
    padding: 0.8rem;
    border-radius: 9px;
}

div[data-testid="stFormSubmitButton"] button {
    min-height: 3rem;
    border-radius: 10px;
    font-weight: 600;
    background: #0e7c86;
    border: none;
    color: white;
}

div[data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.1);
    color: white;
}

/* Example buttons */
.ls-try {
    font-size: 0.9rem;
    opacity: 0.7;
    margin: 1rem 0 0.5rem 0;
}

div[data-testid="stButton"] button {
    border-radius: 999px;
    font-size: 0.85rem;
    padding: 0.2rem 0.7rem;
    border: 1px solid rgba(127,127,127,0.3);
}

/* Verdict card */
.ls-verdict {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    border: 1px solid rgba(127,127,127,0.28);
    border-left: 6px solid var(--tone);
    border-radius: 16px;
    padding: 1.4rem;
    margin: 1.8rem 0 1rem 0;
    background: rgba(127,127,127,0.04);
}

.ls-gauge {
    position: relative;
    width: 132px;
    height: 132px;
    flex: none;
}

.ls-gauge svg {
    transform: rotate(-90deg);
}

.ls-ring-bg {
    stroke: rgba(127,127,127,0.2);
}

.ls-ring {
    stroke: var(--tone);
    stroke-linecap: round;
    animation: ls-fill 1.1s cubic-bezier(.2,.7,.2,1) both;
}

@keyframes ls-fill {
    from {
        stroke-dashoffset: 339.292;
    }
}

@media (prefers-reduced-motion: reduce) {
    .ls-ring {
        animation: none;
    }
}

.ls-gauge-text {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.ls-gauge-num {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.03em;
    line-height: 1;
}

.ls-gauge-label {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 4px;
}

.ls-verdict-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 700;
    font-size: 1.55rem;
    color: var(--tone);
    margin: 0 0 0.4rem 0;
}

.ls-verdict-text {
    opacity: 0.82;
    line-height: 1.55;
    margin: 0;
}

/* URL display */
.ls-url {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 0.9rem;
    border: 1px solid rgba(127,127,127,0.28);
    border-radius: 10px;
    padding: 0.9rem;
    background: rgba(127,127,127,0.06);
    overflow-wrap: anywhere;
    margin-bottom: 1.3rem;
}

.ls-url .dim {
    opacity: 0.55;
}

.ls-url .host {
    font-weight: 700;
    color: #0e9aa5;
}

/* Section headings */
.ls-h {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 1.3rem 0 0.7rem 0;
}

/* URL signals */
.ls-signals {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.65rem;
}

.ls-signal {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    border: 1px solid rgba(127,127,127,0.25);
    border-radius: 11px;
    padding: 0.8rem;
    font-size: 0.9rem;
    line-height: 1.45;
    min-width: 0;
}

.ls-signal .dot {
    flex: none;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 0.35rem;
}

.ls-signal.ok .dot {
    background: #16a36b;
}

.ls-signal.flag .dot {
    background: #dc4035;
}

.ls-signal b {
    display: block;
    font-weight: 600;
    margin-bottom: 2px;
}

.ls-signal span {
    opacity: 0.72;
    overflow-wrap: anywhere;
}

/* Percentage bar */
.ls-split {
    display: flex;
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    margin: 0.7rem 0;
    background: rgba(127,127,127,0.15);
}

.ls-split .l {
    background: #16a36b;
    height: 100%;
}

.ls-split .p {
    background: #dc4035;
    height: 100%;
}

.ls-split-legend {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.85rem;
    opacity: 0.85;
}

.ls-note {
    font-size: 0.82rem;
    opacity: 0.68;
    line-height: 1.5;
    margin-top: 0.9rem;
}

/* Technical details expander */
div[data-testid="stExpander"] {
    border: 1px solid rgba(127,127,127,0.28);
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1.3rem;
}

div[data-testid="stExpander"] summary {
    padding: 0.85rem 1rem;
    font-weight: 600;
    line-height: 1.5;
    gap: 0.7rem;
}

div[data-testid="stExpander"] summary span {
    line-height: 1.5;
    overflow-wrap: anywhere;
}

/* Recent checks */
.ls-history-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    border: 1px solid rgba(127,127,127,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.55rem;
    min-width: 0;
}

.ls-history-url {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.ls-history-result {
    flex: none;
    text-align: right;
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.4;
}

/* Disclaimer */
.ls-disclaimer {
    text-align: center;
    font-size: 0.82rem;
    opacity: 0.65;
    line-height: 1.5;
    margin-top: 2rem;
}

/* Mobile layout */
@media (max-width: 560px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.5rem;
    }

    .ls-verdict {
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 1rem;
    }

    .ls-verdict-title {
        font-size: 1.4rem;
    }

    .ls-history-item {
        flex-direction: column;
        gap: 0.5rem;
    }

    .ls-history-result {
        text-align: left;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# 3. HTML HELPER
# ============================================================

def html(markup: str) -> None:
    """Render HTML without Markdown treating indented lines as code."""
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    scaler = joblib.load(SCALER_PATH)

    return model, tokenizer, scaler


# ============================================================
# 5. URL VALIDATION
# ============================================================

def normalize_url(raw_url):
    url = raw_url.strip()

    if not url:
        raise ValueError("Enter a URL to check.")

    if any(char.isspace() for char in url):
        raise ValueError("Remove the spaces from the URL and try again.")

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS links can be checked.")

    if not parsed.hostname:
        raise ValueError(
            "That link has no domain name. Try something like example.com."
        )

    try:
        parsed.port
    except ValueError:
        raise ValueError("The link has an invalid port number.")

    return url


# ============================================================
# 6. PREDICTION
# ============================================================

def predict_url(url, model, tokenizer, scaler):
    sequences = tokenizer.texts_to_sequences([url])

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    features = extract_features(url)

    feature_values = [
        float(features[column])
        for column in FEATURE_COLUMNS
    ]

    structural = np.asarray(
        [feature_values],
        dtype=np.float32,
    )

    scaled = scaler.transform(structural)

    output = model.predict(
        [padded, scaled],
        verbose=0,
    )

    return float(output[0][0]), features


# ============================================================
# 7. RESULT HELPERS
# ============================================================

def verdict_for(score):
    if score < SAFE_BELOW:
        return (
            "#16a36b",
            "Looks legitimate",
            "The model found few phishing patterns in this link. "
            "That isn't a guarantee the site is safe.",
        )

    if score < DANGER_FROM:
        return (
            "#d98a00",
            "Borderline: verify first",
            "The model is uncertain about this link. "
            "Verify the destination independently before sharing information.",
        )

    return (
        "#dc4035",
        "Likely phishing",
        "This link matches patterns commonly associated with phishing. "
        "Avoid entering passwords or personal details.",
    )


def gauge_html(score, tone):
    offset = RING_CIRCUMFERENCE * (1 - score)

    return f"""
    <div class="ls-gauge">
        <svg width="132" height="132" viewBox="0 0 132 132"
             aria-hidden="true">
            <circle class="ls-ring-bg"
                cx="66" cy="66" r="54"
                fill="none" stroke-width="12"/>
            <circle class="ls-ring"
                cx="66" cy="66" r="54"
                fill="none" stroke="{tone}" stroke-width="12"
                stroke-dasharray="{RING_CIRCUMFERENCE}"
                stroke-dashoffset="{offset:.2f}"/>
        </svg>
        <div class="ls-gauge-text">
            <div class="ls-gauge-num">{score * 100:.1f}%</div>
            <div class="ls-gauge-label">phishing score</div>
        </div>
    </div>
    """


def url_breakdown_html(url):
    parsed = urlparse(url)
    host = parsed.netloc
    rest = url[len(parsed.scheme) + 3 + len(host):]

    return (
        '<div class="ls-url">'
        f'<span class="dim">{html_lib.escape(parsed.scheme)}://</span>'
        f'<span class="host">{html_lib.escape(host)}</span>'
        f'<span class="dim">{html_lib.escape(rest)}</span>'
        "</div>"
    )


# ============================================================
# 8. URL SIGNALS
# ============================================================

def signal_items(f):
    return [
        (
            not f["has_https"],
            "No HTTPS" if not f["has_https"] else "Uses HTTPS",
            "Connection is not encrypted."
            if not f["has_https"]
            else "Connection is encrypted. Scam sites can use HTTPS too.",
        ),
        (
            bool(f["has_ip"]),
            "Raw IP address" if f["has_ip"] else "Domain name present",
            "The link uses an IP address as its hostname."
            if f["has_ip"]
            else "The link uses a domain name.",
        ),
        (
            bool(f["has_at"]),
            "Contains @ symbol" if f["has_at"] else "No @ symbol",
            "The @ symbol can disguise the actual destination."
            if f["has_at"]
            else "No @ symbol was found.",
        ),
        (
            f["url_length"] > 75,
            f"Long link ({int(f['url_length'])} characters)",
            "The URL is relatively long."
            if f["url_length"] > 75
            else "The URL length is within the selected threshold.",
        ),
        (
            f["num_subdomains"] >= 3,
            f"{int(f['num_subdomains'])} subdomain levels",
            "Multiple subdomains deserve closer inspection."
            if f["num_subdomains"] >= 3
            else "Subdomain count is within the selected threshold.",
        ),
        (
            f["num_hyphens"] >= 3 or f["num_digits"] >= 8,
            f"{int(f['num_hyphens'])} hyphens, {int(f['num_digits'])} digits",
            "Many hyphens or digits were found."
            if (f["num_hyphens"] >= 3 or f["num_digits"] >= 8)
            else "Hyphen and digit counts are within the selected thresholds.",
        ),
    ]


def signals_html(features):
    cards = []

    for flagged, title, detail in signal_items(features):
        cls = "flag" if flagged else "ok"

        cards.append(
            f'<div class="ls-signal {cls}">'
            '<div class="dot"></div>'
            f'<div><b>{html_lib.escape(title)}</b>'
            f'<span>{html_lib.escape(detail)}</span></div>'
            '</div>'
        )

    return f'<div class="ls-signals">{"".join(cards)}</div>'


# ============================================================
# 9. EXAMPLE URL BUTTONS
# ============================================================

def set_example(url):
    st.session_state["url_input"] = url
    st.session_state["auto_run"] = True


# ============================================================
# 10. HEADER
# ============================================================

html("""
<h1 class="ls-brand">Link<span>Shield</span></h1>
<p class="ls-tagline">
    Paste a link and see how it scores before you click it.
</p>
""")


# ============================================================
# 11. URL INPUT
# ============================================================

with st.form("url_checker_form"):
    url_input = st.text_input(
        "Website URL",
        key="url_input",
        placeholder="https://example.com/login",
    )

    check_button = st.form_submit_button(
        "Check link",
        type="primary",
        use_container_width=True,
    )

html('<div class="ls-try">Try an example:</div>')

example_cols = st.columns(len(EXAMPLES))

for col, (label, example_url) in zip(example_cols, EXAMPLES.items()):
    col.button(
        label,
        key=f"example_{label}",
        on_click=set_example,
        args=(example_url,),
        use_container_width=True,
    )


# ============================================================
# 12. ANALYSIS
# ============================================================

auto_run = st.session_state.pop("auto_run", False)

if check_button or auto_run:

    try:
        url = normalize_url(
            st.session_state.get("url_input", "")
        )

        with st.spinner("Checking this link..."):
            model, tokenizer, scaler = load_artifacts()

            phishing_score, features = predict_url(
                url,
                model,
                tokenizer,
                scaler,
            )

        phishing_score = min(max(phishing_score, 0.0), 1.0)
        legitimate_score = 1.0 - phishing_score

        tone, title, description = verdict_for(phishing_score)

        # Main result card
        html(f"""
        <div class="ls-verdict">
            {gauge_html(phishing_score, tone)}
            <div>
                <p class="ls-verdict-title" style="color:{tone}">
                    {title}
                </p>
                <p class="ls-verdict-text">{description}</p>
            </div>
        </div>
        """)

        # URL breakdown
        html(url_breakdown_html(url))

        # Two complementary model scores
        html(f"""
        <div class="ls-split" role="img"
             aria-label="Legitimate score {legitimate_score * 100:.1f} percent,
             phishing score {phishing_score * 100:.1f} percent">
            <div class="l"
                 style="width:{legitimate_score * 100:.2f}%"></div>
            <div class="p"
                 style="width:{phishing_score * 100:.2f}%"></div>
        </div>

        <div class="ls-split-legend">
            <span style="color:#16a36b">
                ● Legitimate {legitimate_score * 100:.2f}%
            </span>
            <span style="color:#dc4035">
                ● Phishing {phishing_score * 100:.2f}%
            </span>
        </div>
        """)

        # URL signals
        html('<div class="ls-h">What stood out</div>')
        html(signals_html(features))

        html("""
        <div class="ls-note">
            These URL checks provide additional context.
            They are not separate model predictions.
            The two model scores add up to 100% and do not guarantee
            that a website is safe.
        </div>
        """)

        # Technical details
        with st.expander("Technical Details", expanded=False):
            st.markdown("**Link analysed**")
            st.code(url, language=None)

            st.markdown("**Extracted structural features**")
            st.json({
                column: features[column]
                for column in FEATURE_COLUMNS
            })

        # Recent checks
        history = st.session_state.setdefault("history", [])

        entry = (url, phishing_score)

        if not history or history[0] != entry:
            history.insert(0, entry)

        del history[5:]

    except ValueError as error:
        st.warning(str(error))

    except FileNotFoundError:
        st.error(
            "A model file is missing. Check that the models folder "
            "contains the trained model, tokenizer and scaler."
        )

    except Exception as error:
        st.error(f"Couldn't analyse this link: {error}")
        st.caption("Check the VS Code terminal for the full error.")


# ============================================================
# 13. RECENT CHECKS
# ============================================================

if st.session_state.get("history"):

    html('<div class="ls-h">Recent Checks</div>')

    for past_url, past_score in st.session_state["history"]:

        past_tone, past_title, _ = verdict_for(past_score)

        html(f"""
        <div class="ls-history-item">
            <div class="ls-history-url">
                {html_lib.escape(past_url)}
            </div>
            <div class="ls-history-result" style="color:{past_tone}">
                <b>{past_score * 100:.1f}% ·
                {html_lib.escape(past_title)}</b>
            </div>
        </div>
        """)


# ============================================================
# 14. DISCLAIMER
# ============================================================

html("""
<div class="ls-disclaimer">
    Machine-learning results are informational only.
    Always verify suspicious links independently.
</div>
""")
