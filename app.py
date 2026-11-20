from datetime import date

import pandas as pd
import streamlit as st
from services.environment import (
    anomaly_model_versions,
    assess_coverage,
    calculate_environment_assessment,
    forecast_readiness,
    infer_anomalies,
    infer_forecast,
    list_anomaly_models,
    list_forecast_models,
    load_extracted_files,
    load_active_anomaly_model,
    load_uploaded_files,
    monthly_trend,
    monthly_training_candidates,
    normalize_frames,
    persist_processed_records,
    promote_anomaly_model,
    train_anomaly_model,
    train_forecast_model,
    z_score_anomalies,
)

st.set_page_config(page_title="ESG Assessment Platform", page_icon="🌿", layout="wide")

# --------------------------------------------------------------------------
# State + language helper
# --------------------------------------------------------------------------
DEFAULTS = {
    "lang": "en",
    "page": "home",
    "project": {"name": "", "desc": "", "location": "", "duration": "", "category": 0},
    "section": 0,
    "answers": {},          # {question_index: option_index}
    "community_scores": None,
    "ent_scores": None,     # {"overall","E","S","G","file","date"}
    "ent_file": None,
    "ent_records": None,
    "ent_filename": None,
    "ent_diagnostics": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def tr(en, vi):
    """Return the Vietnamese text when the language toggle is on VI, otherwise English."""
    return vi if st.session_state.lang == "vi" else en


def go(page):
    st.session_state.page = page


def toggle_lang():
    st.session_state.lang = "vi" if st.session_state.lang == "en" else "en"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
DIM_COLORS = ["#21785A", "#2B7BAE", "#8E7438"]
DIM_KEYS = "ESG"


def dim_names():
    return [tr("Environmental", "Môi trường"), tr("Social", "Xã hội"),
            tr("Governance & Economic", "Quản trị & Kinh tế")]


def short_dims():
    return [tr("environmental", "môi trường"), tr("social", "xã hội"),
            tr("governance and economic", "quản trị và kinh tế")]


def section_names():
    return [tr("Environmental", "Môi trường"), tr("Social", "Xã hội"),
            tr("Governance & Economic", "Quản trị và Kinh tế")]


QUESTIONS_VI = [
    # ---- Môi trường (1-5)
    ("Dự án có biện pháp bảo vệ đa dạng sinh học và các khu vực cây xanh tại địa phương không?",
     ["Chưa có biện pháp", "Đã có kế hoạch thực hiện", "Đã triển khai và theo dõi kết quả"]),
    ("Dự án quản lý năng lượng và giảm phát thải khí nhà kính như thế nào?",
     ["Chưa có hoạt động cụ thể", "Có hoạt động tiết kiệm năng lượng", "Có mục tiêu giảm phát thải và đo lường định kỳ"]),
    ("Dự án có chính sách và hệ thống quản lý môi trường, đồng thời quản lý trách nhiệm đối với sản phẩm và dịch vụ không?",
     ["Chưa có chính sách", "Có cam kết hoặc hướng dẫn cơ bản", "Có chính sách, người phụ trách và đánh giá định kỳ"]),
    ("Dự án quản lý nước, chất thải và các nguồn gây ô nhiễm như thế nào?",
     ["Chưa có biện pháp", "Có một số hoạt động tiết kiệm nước, phân loại hoặc giảm thải", "Có mục tiêu, quy trình và đo lường định kỳ"]),
    ("Dự án có sử dụng nguyên vật liệu bền vững, bao bì thân thiện môi trường và thực hành lâm nghiệp bền vững không?",
     ["Chưa xem xét", "Có ưu tiên một số vật liệu hoặc bao bì bền vững", "Có tiêu chí rõ ràng và theo dõi việc sử dụng"]),
    # ---- Xã hội (6-10)
    ("Cộng đồng địa phương tham gia vào quá trình xây dựng và triển khai dự án như thế nào?",
     ["Chưa được tham vấn", "Được lấy ý kiến khi bắt đầu", "Tham gia thường xuyên và có kênh phản hồi"]),
    ("Dự án có hỗ trợ sức khỏe, dinh dưỡng hoặc khả năng tiếp cận dịch vụ thiết yếu không?",
     ["Chưa có hoạt động hỗ trợ", "Có một số hoạt động hỗ trợ", "Có chương trình hỗ trợ và theo dõi người hưởng lợi"]),
    ("Dự án bảo đảm quyền con người, điều kiện lao động và cơ hội bình đẳng ở mức độ nào?",
     ["Chưa có hướng dẫn", "Có cam kết hoặc hướng dẫn cơ bản", "Có quy định rõ ràng và cơ chế tiếp nhận phản ánh"]),
    ("Dự án quản lý an toàn lao động và an toàn di chuyển cho người tham gia như thế nào?",
     ["Chưa đánh giá rủi ro", "Có hướng dẫn an toàn cơ bản", "Có tập huấn, đánh giá rủi ro và theo dõi sự cố"]),
    ("Dự án có bảo vệ dữ liệu cá nhân và hỗ trợ phát triển năng lực cho người tham gia không?",
     ["Chưa có biện pháp", "Có thực hiện một phần", "Có quy trình bảo mật và đào tạo thường xuyên"]),
    # ---- Quản trị và Kinh tế (11-15)
    ("Dự án có phân công trách nhiệm, quy tắc đạo đức và công khai các quyết định quan trọng không?",
     ["Chưa có quy định rõ ràng", "Có phân công trách nhiệm cơ bản", "Có quy định, giám sát và công khai thông tin"]),
    ("Dự án quản lý nguồn tài trợ, chi phí và các nghĩa vụ thuế như thế nào?",
     ["Chưa theo dõi có hệ thống", "Có ghi chép thu chi", "Có báo cáo tài chính minh bạch và kiểm tra định kỳ"]),
    ("Dự án có quy trình quản lý rủi ro, chất lượng hoạt động và an toàn thông tin không?",
     ["Chưa có quy trình", "Có nhận diện một số rủi ro", "Có quy trình, người phụ trách và phương án ứng phó"]),
    ("Dự án lựa chọn đối tác và nhà cung cấp dựa trên tiêu chí bền vững ở mức độ nào?",
     ["Chưa có tiêu chí", "Có xem xét một số tiêu chí", "Có tiêu chí rõ ràng và đánh giá định kỳ"]),
    ("Dự án có xác định vấn đề trọng yếu, đổi mới và báo cáo tác động dài hạn không?",
     ["Chưa thực hiện", "Có kế hoạch hoặc báo cáo cơ bản", "Có đánh giá trọng yếu, cải tiến và báo cáo định kỳ"]),
]

QUESTIONS_EN = [
    # ---- Environmental (1-5)
    ("Does the project have measures to protect biodiversity and local green spaces?",
     ["No measures yet", "Plans are in place", "Implemented, with results monitored"]),
    ("How does the project manage energy and reduce greenhouse gas emissions?",
     ["No specific activities yet", "Some energy-saving activities", "Emission-reduction targets with regular measurement"]),
    ("Does the project have an environmental policy and management system, and manage its responsibility for products and services?",
     ["No policy yet", "Basic commitment or guidelines", "Policy, assigned owner and periodic review"]),
    ("How does the project manage water, waste and sources of pollution?",
     ["No measures yet", "Some activities to save water, sort or reduce waste", "Targets, procedures and regular measurement"]),
    ("Does the project use sustainable raw materials, eco-friendly packaging and sustainable forestry practices?",
     ["Not considered yet", "Some sustainable materials or packaging are prioritized", "Clear criteria and usage is tracked"]),
    # ---- Social (6-10)
    ("How does the local community take part in designing and delivering the project?",
     ["Not consulted", "Consulted at the start", "Regular participation with a feedback channel"]),
    ("Does the project support health, nutrition or access to essential services?",
     ["No support activities yet", "Some support activities", "A support programme with beneficiaries tracked"]),
    ("To what extent does the project ensure human rights, working conditions and equal opportunities?",
     ["No guidelines yet", "Basic commitment or guidelines", "Clear rules and a grievance mechanism"]),
    ("How does the project manage occupational safety and travel safety for participants?",
     ["No risk assessment yet", "Basic safety guidelines", "Training, risk assessment and incident tracking"]),
    ("Does the project protect personal data and support capacity building for participants?",
     ["No measures yet", "Partially implemented", "Data-protection procedures and regular training"]),
    # ---- Governance & Economic (11-15)
    ("Does the project assign responsibilities, set ethical rules and publicly disclose key decisions?",
     ["No clear rules yet", "Basic assignment of responsibilities", "Rules, oversight and public disclosure"]),
    ("How does the project manage funding, costs and tax obligations?",
     ["No systematic tracking", "Income and expenses are recorded", "Transparent financial reporting with periodic checks"]),
    ("Does the project have processes for risk management, operational quality and information security?",
     ["No process yet", "Some risks are identified", "Process, owner and response plan in place"]),
    ("To what extent does the project select partners and suppliers on sustainability criteria?",
     ["No criteria yet", "Some criteria considered", "Clear criteria with periodic evaluation"]),
    ("Does the project identify material issues, innovate and report long-term impact?",
     ["Not yet", "Basic plan or report", "Materiality assessment, improvements and regular reporting"]),
]

# Full ESG criteria (35) and the survey question each one is covered by (1-based question number)
CRITERIA = {
    "Environmental": {
        1: ["Biodiversity"],
        2: ["Climate Strategy", "Decarbonization Strategy", "Energy"],
        3: ["Environmental Policy & Management", "Product Stewardship"],
        4: ["Water", "Waste & Pollutants"],
        5: ["Packaging", "Sustainable Raw Materials", "Sustainable Forestry Practices"],
    },
    "Social": {
        6: ["Community Relations"],
        7: ["Contribution to Societal Healthcare", "Health & Nutrition", "Financial Inclusion", "Customer Relations"],
        8: ["Human Rights", "Labor Practices"],
        9: ["Occupational Health & Safety", "Transportation Safety"],
        10: ["Privacy Protection", "Human Capital Management"],
    },
    "Governance & Economic": {
        11: ["Corporate Governance", "Business Ethics", "Policy Influence"],
        12: ["Tax Strategy", "Sustainable Finance"],
        13: ["Risk & Crisis Management", "Product Quality & Recall Management",
             "Information Security / Cybersecurity & System Availability"],
        14: ["Supply Chain Management"],
        15: ["Materiality", "Innovation Management", "Market Opportunities", "Transparency & Reporting"],
    },
}
CRITERIA_KEYS = ["Environmental", "Social", "Governance & Economic"]


def categories():
    return [tr("Environmental conservation", "Bảo tồn môi trường"),
            tr("Community development", "Phát triển cộng đồng"),
            tr("Education & training", "Giáo dục và đào tạo"),
            tr("Health & well-being", "Sức khỏe và phúc lợi"),
            tr("Circular economy & waste", "Kinh tế tuần hoàn và chất thải"),
            tr("Other", "Khác")]


def focus_hints(enterprise=False):
    if enterprise:
        return {"E": tr("Track energy consumption, emissions and water use.",
                        "Theo dõi mức tiêu thụ năng lượng, phát thải và sử dụng nước."),
                "S": tr("Track workforce wellbeing and community outcomes.",
                        "Theo dõi phúc lợi người lao động và kết quả đối với cộng đồng."),
                "G": tr("Strengthen risk management and transparency in reporting.",
                        "Tăng cường quản lý rủi ro và minh bạch trong báo cáo.")}
    return {"E": tr("Track energy consumption, emissions and water use.",
                    "Theo dõi mức tiêu thụ năng lượng, phát thải và sử dụng nước."),
            "S": tr("Track community feedback and participation outcomes.",
                    "Theo dõi phản hồi của cộng đồng và kết quả tham gia."),
            "G": tr("Publish a regular project impact and spending summary.",
                    "Công bố định kỳ báo cáo tác động và chi tiêu của dự án.")}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def community_scores():
    ans = st.session_state.answers
    dims = []
    for d in range(3):
        pts = sum(ans.get(i, 0) for i in range(d * 5, d * 5 + 5))
        dims.append(round(pts / 10 * 100))
    overall = round(sum(dims) / 3)
    return {"overall": overall, "E": dims[0], "S": dims[1], "G": dims[2]}


def headline(score):
    if score >= 75:
        return tr("A strong foundation", "Một nền tảng vững chắc")
    if score >= 50:
        return tr("A solid start", "Một khởi đầu tốt")
    return tr("Room to grow", "Còn nhiều dư địa để phát triển")


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--dark:#183F36;--green:#21785A;--muted:#62726C;--pale:#E9F2EA;--line:#E1E9E5;--bg:#F6F8F7;}
html, body, .stApp, [class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg);overflow-x:hidden;}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none!important;}
.block-container{max-width:1120px;padding:0 24px 0 24px!important;}
h1,h2,h3,h4{color:var(--dark);font-family:'Inter',sans-serif;letter-spacing:-0.02em;}
p, label, span, div{color:inherit;}

/* header (full-bleed white bar) */
.st-key-header{position:relative;padding:14px 0;margin-bottom:28px;}
.st-key-header::before{content:"";position:absolute;top:0;bottom:0;left:-100vw;right:-100vw;background:#fff;border-bottom:1px solid var(--line);z-index:0;}
.st-key-header > div{position:relative;z-index=1;align-items:center;}
.brand{display:flex;align-items:center;gap:14px;}
.brand .t{font-weight:600;font-size:17px;color:var(--dark);line-height:1.15;}
.brand .s{font-size:9px;letter-spacing:.06em;color:var(--muted);font-weight:600;}
.st-key-nav_home button, .st-key-nav_meth button{border:none!important;background:transparent!important;color:var(--muted)!important;box-shadow:none!important;padding:0 4px!important;font-weight:400!important;}
.st-key-nav_home button:hover, .st-key-nav_meth button:hover{color:var(--dark)!important;}

/* buttons */
.stButton>button, .stDownloadButton>button{border-radius:8px;border:1px solid var(--line);background:#fff;color:var(--dark);font-weight:600;font-size:14px;padding:10px 20px;min-height:44px;font-family:'Inter',sans-serif;}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--green);color:var(--green);}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{background:var(--green);border-color:var(--green);color:#fff;}
.stButton>button[kind="primary"]:hover{background:#1b6149;color:#fff;}
.stButton>button[kind="primary"] p, .stDownloadButton>button[kind="primary"] p{color:#fff!important;}
.stButton>button:disabled{opacity:.5;}

/* cards */
[class*="st-key-card_"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;}
.pill{display:inline-block;background:var(--pale);color:#1F6F52;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:7px 12px;border-radius:999px;}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#1F6F52;margin:6px 0 8px;}
.eyebrow-sm{font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);}
.h1{font-size:58px;line-height:1.18;font-weight:700;color:var(--dark);margin:18px 0 22px;}
.h2{font-size:34px;line-height:1.2;font-weight:700;color:var(--dark);margin:6px 0 14px;}
.h3{font-size:26px;font-weight:600;color:var(--dark);margin:10px 0 10px;}
.lead{font-size:17px;color:var(--muted);line-height:1.6;}
.muted{color:var(--muted);font-size:14px;line-height:1.55;}
.small{color:var(--muted);font-size:12px;}
.num{color:var(--green);font-weight:600;margin-right:10px;}

/* hero score card */
.score-dark{background:var(--dark);border-radius:22px;padding:34px 32px;color:#fff;}
.score-dark .lab{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#B9D3C7;font-weight:500;}
.score-dark .big{font-size:84px;font-weight:700;line-height:1;margin:28px 0 22px;color:#fff;}
.score-dark .big span{font-size:24px;font-weight:400;color:#B9D3C7;margin-left:10px;}
.score-dark .cap{font-size:19px;font-weight:600;color:#fff;margin-bottom:18px;}
.score-dark .txt{font-size:14px;color:#CFE1D8;line-height:1.55;}
.hrow{display:flex;justify-content:space-between;font-size:13px;color:#fff;margin-top:14px;}
.htrack{height:6px;border-radius:6px;background:rgba(255,255,255,.12);margin-top:6px;}
.hfill{height:6px;border-radius:6px;}
.score-dark .foot{font-size:11px;color:#B9D3C7;margin-top:22px;}

/* bars */
.brow{display:flex;justify-content:space-between;font-size:15px;color:var(--dark);margin-top:18px;}
.brow b{font-weight:600;}
.btrack{height:12px;border-radius:8px;background:#EEF1EF;margin-top:8px;}
.bfill{height:12px;border-radius:8px;}
.kpi{background:var(--pale);border-radius:12px;padding:18px 20px;}
.kpi .l{font-size:13px;color:var(--muted);}
.kpi .v{font-size:36px;font-weight:700;line-height:1.2;margin-top:10px;}
.kpi .v span{font-size:13px;font-weight:400;color:var(--muted);margin-left:6px;}
.note{background:#EAF3FB;color:#2A6A97;font-size:12.5px;padding:12px 16px;border-radius:8px;margin:14px 0 22px;}
.hint{font-size:12px;color:#2B7BAE;margin-top:14px;line-height:1.5;}

/* side info panel */
.side{background:var(--pale);border:1px solid var(--line);border-radius:14px;padding:30px;}
.side h3{font-size:25px;line-height:1.25;margin:0 0 20px;}
.side .st{font-weight:600;color:var(--dark);margin:18px 0 6px;font-size:14.5px;}
.side .sd{color:var(--muted);font-size:13px;line-height:1.5;}

/* step tabs */
.steps{display:flex;justify-content:space-between;margin:14px 0 30px;font-size:13px;font-weight:600;color:var(--muted);}
.steps .on{color:#1F6F52;}
.steps>div{flex:1;}

/* inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"]>div{background:#fff;border:1px solid var(--line);border-radius:8px;color:var(--dark);font-size:14.5px;}
.stTextInput label p, .stTextArea label p, .stSelectbox label p{color:var(--dark);font-weight:500;font-size:13.5px;}

/* radio as option cards */
[data-testid="stRadio"] [role="radiogroup"]{gap:10px;}
[data-testid="stRadio"] label{background:#F7FAF8;border-radius:8px;padding:14px 16px!important;width:100%;margin:0;}
[data-testid="stRadio"] label:has(input:checked){background:var(--pale);}
[data-testid="stRadio"] label p{color:var(--muted);font-size:14.5px;}
[data-testid="stRadio"] label:has(input:checked) p{color:var(--dark);}
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p{color:var(--dark)!important;font-weight:600;font-size:16px;margin-bottom:8px;}

/* progress */
.ptrack{height:6px;background:var(--pale);border-radius:6px;margin-top:10px;}
.pfill{height:6px;background:var(--green);border-radius:6px;}
.sec-card{border-radius:10px;padding:16px 18px;margin-bottom:10px;background:#fff;}
.sec-card.cur{background:var(--pale);}
.sec-card .a{font-weight:600;color:var(--muted);font-size:15px;}
.sec-card.cur .a{color:#1F6F52;}
.sec-card .b{font-size:12px;color:var(--muted);margin-top:4px;}

/* login hero */
.dark-panel{background:var(--dark);border-radius:20px;padding:44px 40px;color:#fff;min-height:480px;}
.dark-panel .e{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#B9D3C7;}
.dark-panel .h{font-size:38px;font-weight:700;line-height:1.25;margin:26px 0 22px;color:#fff;}
.dark-panel .p{font-size:16px;color:#D3E4DB;line-height:1.55;margin-bottom:22px;}
.dark-panel .s{font-weight:600;margin-top:16px;color:#fff;}
.dark-panel .d{font-size:13px;color:#B9D3C7;margin-top:4px;}
.banner{background:var(--dark);border-radius:14px;padding:30px 30px;color:#fff;display:flex;justify-content:space-between;align-items:center;}
.banner .h{font-size:24px;font-weight:600;color:#fff;}
.banner .p{font-size:14px;color:#CFE1D8;margin-top:8px;}
.banner .f{text-align:right;font-weight:700;font-size:13px;letter-spacing:.03em;color:#fff;}
.banner .f small{display:block;font-weight:400;color:#B9D3C7;margin-top:6px;}

/* upload */
[data-testid="stFileUploader"] section{border:2px dashed #BFDCCB;background:#F5F9F6;border-radius:12px;padding:44px 20px;}
.filechip{background:var(--pale);border-radius:10px;padding:14px 18px;display:flex;justify-content:space-between;margin-top:14px;}
.filechip b{color:var(--dark);font-size:14px;}
.filechip .r{color:#1F6F52;font-size:12px;font-weight:600;}

/* pipeline */
.pstep{display:flex;justify-content:space-between;padding:16px 14px;border-radius:10px;margin-bottom:6px;}
.pstep.act{background:#EAF3FB;}
.pstep b{color:var(--dark);font-size:15px;}
.pstep .d{color:var(--muted);font-size:12px;margin-top:3px;}
.pstep .st{font-size:12px;color:#1F6F52;}
.pstep.act .st{color:#2B7BAE;}
.pstep.todo .st{color:var(--muted);}

/* footer */
.st-key-footer{position:relative;margin-top:60px;padding:26px 0;}
.st-key-footer::before{content:"";position:absolute;top:0;bottom:0;left:-100vw;right:-100vw;background:#fff;border-top:1px solid var(--line);z-index:0;}
.st-key-footer > div{position:relative;z-index=1;}
.foot-l b{font-size:12px;color:var(--dark);}
.foot-l div{font-size:11px;color:var(--muted);}
.foot-r{text-align:right;font-size:11px;color:#2B7BAE;}
@media(max-width:800px){.h1{font-size:40px}.score-dark .big{font-size:64px}}
.st-key-header button{white-space:nowrap;}
.st-key-lang_btn button{min-height:38px;font-size:13px;padding:6px 10px;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

LOGO_SVG = """
<svg width="76" height="76" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
<defs><clipPath id="c"><circle cx="28" cy="28" r="27"/></clipPath></defs>
<g clip-path="url(#c)"><rect width="56" height="56" fill="#fff"/>
<path d="M2 20 Q16 2 34 8 Q28 24 10 28 Z" fill="#7FB241"/>
<path d="M30 6 Q50 4 54 22 Q38 24 30 6Z" fill="#9CC95B"/>
<path d="M0 40 Q24 28 56 38 L56 56 L0 56Z" fill="#5E9E3A"/>
<path d="M8 50 Q30 36 56 46 L56 56 L8 56Z" fill="#8DBF4E"/>
<path d="M44 40 Q50 36 54 38 Q50 44 44 44Z" fill="#E8A020"/></g>
<text x="28" y="33" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" font-weight="500" fill="#2A7F5B">E<tspan fill="#2B7BAE">S</tspan><tspan fill="#E8A020">G</tspan></text>
</svg>"""


def _load_logo():
    """Dùng ảnh logo thật nếu có file assets/logo.(png|jpg|jpeg|webp|svg); nếu không dùng logo SVG dự phòng."""
    import base64
    from pathlib import Path
    base = Path(__file__).parent / "assets"
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".svg": "image/svg+xml"}
    for ext, m in mime.items():
        f = base / f"logo{ext}"
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode()
            return f'<img src="data:{m};base64,{b64}" style="height:76px;width:auto;display:block" alt="ESG logo">'
    return LOGO_SVG


def _load_logo():
    """Use the real logo image if assets/logo.(png|jpg|jpeg|webp|svg) exists; otherwise the fallback SVG."""
    import base64
    from pathlib import Path
    base = Path(__file__).parent / "assets"
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".svg": "image/svg+xml"}
    for ext, m in mime.items():
        f = base / f"logo{ext}"
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode()
            return f'<img src="data:{m};base64,{b64}" style="height:56px;width:auto;display:block" alt="ESG logo">'
    return LOGO_SVG


LOGO = _load_logo()


# --------------------------------------------------------------------------
# Shared components
# --------------------------------------------------------------------------
def header():
    page = st.session_state.page
    with st.container(key="header"):
        c1, c2, c3, c4, c5 = st.columns([3.6, 0.9, 1.9, 1.5, 2.3], vertical_alignment="center")
        c1.markdown(
            f'<div class="brand">{LOGO}<div><div class="t">ESG Assessment Platform</div>'
            f'<div class="s">BY DATA4ESGENIUS</div></div></div>', unsafe_allow_html=True)
        c2.button(tr("Home", "Trang chủ"), key="nav_home", on_click=go, args=("home",))
        c3.button(tr("Our methodology", "Phương pháp luận"), key="nav_meth", on_click=go, args=("methodology",))
        c4.button(tr("🌐 Tiếng Việt", "🌐 English"), key="lang_btn", on_click=toggle_lang, use_container_width=True)
        if page == "home":
            c5.button(tr("Enterprise Login", "Đăng nhập doanh nghiệp"), key="hdr_cta", on_click=go,
                      args=("ent_login",), use_container_width=True)
        else:
            c5.button(tr("Back to Home", "Về trang chủ"), key="hdr_cta", on_click=go, args=("home",),
                      use_container_width=True)


def footer():
    with st.container(key="footer"):
        a, b = st.columns(2)
        a.markdown(f'<div class="foot-l"><b>Data4ESGenius Team</b><div>ESG Evaluation Tool © 2025</div></div>',
                   unsafe_allow_html=True)
        b.markdown('<div class="foot-r">Data4ESGenius · data4esgenius@gmail.com</div>', unsafe_allow_html=True)


def disclaimer():
    return tr("Scores are indicative, not a certification or an official S&amp;P Global ESG Score.",
              "Điểm số mang tính tham khảo, không phải chứng nhận hay điểm ESG chính thức của S&amp;P Global.")


def bars(scores, colors=DIM_COLORS):
    html = ""
    for name, key, col in zip(dim_names(), DIM_KEYS, colors):
        v = scores[key]
        html += (f'<div class="brow"><span>{name}</span><b style="color:{col}">{v}</b></div>'
                 f'<div class="btrack"><div class="bfill" style="width:{v}%;background:{col}"></div></div>')
    return html


def report_html(title, subtitle, s):
    rows = "".join(f"<tr><td>{n}</td><td><b>{s[k]}</b> / 100</td></tr>" for n, k in zip(dim_names(), DIM_KEYS))
    lang = st.session_state.lang
    return f"""<html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Inter,Arial,sans-serif;max-width:720px;margin:40px auto;color:#183F36}}
td{{padding:8px 16px;border-bottom:1px solid #E1E9E5}}.big{{font-size:56px;font-weight:700}}</style></head><body>
<h1>{title}</h1><p>{subtitle} · {date.today():%d %b %Y}</p>
<div class="big">{s['overall']} / 100</div><p>{headline(s['overall'])}</p>
<table>{rows}</table>
<p style="color:#62726C;font-size:12px">{disclaimer()}</p>
</body></html>"""


def column_chart_svg(s):
    labels = [tr("Environmental", "Môi trường"), tr("Social", "Xã hội"), tr("Gov. & Econ.", "Quản trị & KT")]
    left, top, bottom, width = 40, 20, 210, 440
    h = bottom - top
    out = [f'<svg viewBox="0 0 {width} 252" width="100%" role="img" style="font-family:Inter,sans-serif">']
    for g in (0, 25, 50, 75, 100):
        y = bottom - h * g / 100
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-10}" y2="{y:.1f}" stroke="#E1E9E5" stroke-width="1"/>')
        out.append(f'<text x="{left-8}" y="{y+4:.1f}" font-size="11" fill="#62726C" text-anchor="end">{g}</text>')
    slot = (width - 10 - left) / 3
    bw = 64
    for i, (k, col, lab) in enumerate(zip(DIM_KEYS, DIM_COLORS, labels)):
        v = s[k]
        x = left + slot * i + (slot - bw) / 2
        bh = h * v / 100
        y = bottom - bh
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{bh:.1f}" rx="6" fill="{col}"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" font-size="14" font-weight="700" fill="{col}" text-anchor="middle">{v}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{bottom+22}" font-size="12" fill="#183F36" text-anchor="middle">{lab}</text>')
    out.append('</svg>')
    return "".join(out)


def pie_chart_svg(s):
    import math
    vals = [s[k] for k in DIM_KEYS]
    total = sum(vals)
    names = dim_names()
    cx = cy = 100
    r = 88
    out = ['<div style="display:flex;align-items:center;gap:22px;flex-wrap:wrap;justify-content:center">',
           '<svg viewBox="0 0 200 200" width="210" height="210" role="img" style="font-family:Inter,sans-serif">']
    if total <= 0:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#EEF1EF"/>')
    else:
        ang = -math.pi / 2
        for v, col in zip(vals, DIM_COLORS):
            if v <= 0:
                continue
            frac = v / total
            if frac >= 0.9999:
                out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}"/>')
                mid = ang
            else:
                a2 = ang + frac * 2 * math.pi
                x1, y1 = cx + r * math.cos(ang), cy + r * math.sin(ang)
                x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
                large = 1 if frac > 0.5 else 0
                out.append(f'<path d="M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {large} 1 {x2:.2f},{y2:.2f} Z" fill="{col}" stroke="#fff" stroke-width="2"/>')
                mid = ang + frac * math.pi
                ang = a2
            lx, ly = cx + r * 0.62 * math.cos(mid), cy + r * 0.62 * math.sin(mid)
            out.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">{round(frac*100)}%</text>')
    out.append('</svg><div>')
    for n, v, col in zip(names, vals, DIM_COLORS):
        out.append(f'<div style="display:flex;align-items:center;gap:10px;margin:9px 0;font-size:14px;color:var(--dark)">'
                   f'<span style="width:12px;height:12px;border-radius:3px;background:{col};display:inline-block"></span>'
                   f'<span>{n}</span><b style="margin-left:6px">{v}</b></div>')
    out.append('</div></div>')
    return "".join(out)


def results_view(*, step_label, title_sub, note, s, tell, hints, source, back_label, back_page, report_name):
    st.markdown(f'<div class="eyebrow">{step_label}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1.3], vertical_alignment="bottom")
    c1.markdown(f'<div class="h2" style="margin-bottom:4px">{tr("Your sustainability, measured.", "Mức độ bền vững của bạn, được đo lường.")}</div>'
                f'<div class="lead" style="font-size:16px">{title_sub}</div>', unsafe_allow_html=True)
    c2.download_button(tr("Download Report", "Tải báo cáo"),
                       data=report_html(tr("ESG Assessment Report", "Báo cáo đánh giá ESG"), title_sub, s),
                       file_name=report_name, mime="text/html", type="primary", use_container_width=True)
    st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

    a, b = st.columns([1, 2], gap="medium")
    a.markdown(f'<div class="score-dark" style="min-height:286px"><div class="lab">{tr("Overall ESG score", "Điểm ESG tổng thể")}</div>'
               f'<div class="big">{s["overall"]}<span>/ 100</span></div>'
               f'<div class="cap">{headline(s["overall"])}</div>'
               f'<div class="txt">{tr("Build on your strengths and focus your next steps where they matter most.", "Phát huy điểm mạnh và tập trung các bước tiếp theo vào những nơi quan trọng nhất.")}</div></div>',
               unsafe_allow_html=True)
    with b.container(key="card_1"):
        st.markdown(f'<div class="h3" style="margin-top:6px">{tr("Performance across E, S &amp; G", "Kết quả theo các trụ cột E, S &amp; G")}</div>'
                    f'<div class="eyebrow-sm" style="margin-top:14px">{tr("SCORE · 0–100", "ĐIỂM · 0–100")}</div>{bars(s)}'
                    f'<div class="small" style="margin-top:18px">{tr("All dimension scores are shown on a 0–100 scale.", "Điểm của tất cả các khía cạnh được hiển thị theo thang 0–100.")}</div>',
                    unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="medium")
    with ch1.container(key="card_c1"):
        st.markdown(f'<div class="h3" style="font-size:21px;margin-top:0">{tr("Score by dimension", "Điểm theo từng khía cạnh")}</div>'
                    f'<div class="small" style="margin-bottom:8px">{tr("Column chart · 0–100", "Biểu đồ cột · 0–100")}</div>'
                    f'{column_chart_svg(s)}', unsafe_allow_html=True)
    with ch2.container(key="card_c2"):
        st.markdown(f'<div class="h3" style="font-size:21px;margin-top:0">{tr("Score composition", "Cơ cấu điểm số")}</div>'
                    f'<div class="small" style="margin-bottom:14px">{tr("Pie chart · each dimension’s share of the three scores combined", "Biểu đồ tròn · tỷ trọng của từng khía cạnh trong tổng ba điểm")}</div>'
                    f'{pie_chart_svg(s)}', unsafe_allow_html=True)

    d, e = st.columns(2, gap="medium")
    with d.container(key="card_2"):
        st.markdown(f'<div class="h3" style="font-size:21px">{tr("What your results tell you", "Kết quả cho bạn biết điều gì")}</div>'
                    f'<div class="muted" style="font-size:15px">{tell}</div>'
                    f'<div class="small" style="margin-top:18px">{source}</div>', unsafe_allow_html=True)
    names = dict(zip(DIM_KEYS, dim_names()))
    order = sorted(DIM_KEYS, key=lambda k: s[k])[:2]
    items = "".join(
        f'<div style="margin-top:14px"><b style="color:var(--dark)">0{i+1}&nbsp; {names[k]} · {s[k]}</b>'
        f'<div class="muted" style="margin-top:8px">{hints[k]}</div></div>'
        for i, k in enumerate(order))
    with e.container(key="card_3"):
        st.markdown(f'<div class="h3" style="font-size:21px">{tr("Where to focus next", "Nên tập trung vào đâu tiếp theo")}</div>{items}',
                    unsafe_allow_html=True)

    st.write("")
    st.button(back_label, on_click=go, args=(back_page,))
    st.markdown(f'<div class="small" style="margin-top:20px">{disclaimer()}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home():
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(f'<span class="pill">{tr("A clearer path to sustainability", "Con đường rõ ràng hơn tới bền vững")}</span>'
                    f'<div class="h1">{tr("Measure Your<br>Sustainability<br>Impact", "Đo lường<br>tác động bền vững<br>của bạn")}</div>'
                    f'<div class="lead" style="max-width:470px">{tr("Turn your environmental, social and governance efforts into a clearer picture of your impact.", "Biến các nỗ lực về môi trường, xã hội và quản trị của bạn thành bức tranh tác động rõ ràng hơn.")}</div>',
                    unsafe_allow_html=True)
        st.write("")
        b1, b2, _ = st.columns([1.4, 1.6, 0.6])
        b1.button(tr("Start Assessment →", "Bắt đầu đánh giá →"), type="primary", key="h_start", on_click=go,
                  args=("c_info",), use_container_width=True)
        b2.button(tr("Enterprise Login", "Đăng nhập doanh nghiệp"), key="h_ent", on_click=go, args=("ent_login",),
                  use_container_width=True)
        st.markdown(f'<div class="small" style="margin-top:14px">{tr("For community projects and enterprises.", "Dành cho dự án cộng đồng và doanh nghiệp.")}</div>',
                    unsafe_allow_html=True)
    with right:
        rows = ""
        for n, v, c in zip(dim_names(), (82, 76, 76), ("#A8D58A", "#8CC8E8", "#D6E5A0")):
            rows += (f'<div class="hrow"><span>{n}</span><b>{v}</b></div>'
                     f'<div class="htrack"><div class="hfill" style="width:{v}%;background:{c}"></div></div>')
        st.markdown(f'<div class="score-dark"><div class="lab">{tr("Your impact, at a glance", "Tác động của bạn trong nháy mắt")}</div>'
                    f'<div class="big" style="margin:34px 0 26px">78<span>/ 100</span></div>'
                    f'<div class="cap">{tr("A strong foundation for lasting impact.", "Một nền tảng vững chắc cho tác động lâu dài.")}</div>{rows}'
                    f'<div class="foot">{tr("Illustrative assessment · Example data", "Đánh giá minh họa · Dữ liệu mẫu")}</div></div>',
                    unsafe_allow_html=True)

    st.markdown(f'<div class="h3" style="margin-top:38px">{tr("One platform. Two ways to assess.", "Một nền tảng. Hai cách đánh giá.")}</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a.container(key="card_4"):
        st.markdown(f'<div class="eyebrow-sm"><span class="num">01</span>{tr("GUEST · NO LOGIN NEEDED", "KHÁCH · KHÔNG CẦN ĐĂNG NHẬP")}</div>'
                    f'<div class="h3">{tr("Community Project", "Dự án cộng đồng")}</div>'
                    f'<div class="muted">{tr("Assess your community project through a guided ESG survey.", "Đánh giá dự án cộng đồng của bạn qua khảo sát ESG có hướng dẫn.")}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.button(tr("Start Assessment →", "Bắt đầu đánh giá →"), type="primary", key="card_start", on_click=go,
                  args=("c_info",))
    with b.container(key="card_5"):
        st.markdown(f'<div class="eyebrow-sm"><span class="num">02</span>{tr("ENTERPRISE · LOGIN REQUIRED", "DOANH NGHIỆP · YÊU CẦU ĐĂNG NHẬP")}</div>'
                    f'<div class="h3">{tr("Enterprise Assessment", "Đánh giá doanh nghiệp")}</div>'
                    f'<div class="muted">{tr("Upload company data to review Environment indicators, coverage and trends.", "Tải dữ liệu doanh nghiệp để xem chỉ số Environment, độ bao phủ và xu hướng.")}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.button(tr("Enterprise Login →", "Đăng nhập doanh nghiệp →"), key="card_ent", on_click=go,
                  args=("ent_login",))


def page_methodology():
    st.markdown("""<style>
    .mt{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:10px}
    .mt th{text-align:left;color:var(--dark);background:var(--pale);padding:10px 12px;font-weight:600}
    .mt td{padding:10px 12px;border-bottom:1px solid var(--line);color:var(--muted);vertical-align:top}
    .mt td b{color:var(--dark)}
    .tag{display:inline-block;font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px}
    .t-high{background:#FBE9E7;color:#B4432F}.t-med{background:#FFF4DB;color:#8A6410}.t-low{background:#E9F2EA;color:#1F6F52}
    .ref{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:8px}
    </style>""", unsafe_allow_html=True)
    D = 'style="color:var(--dark)"'

    st.markdown(f'<div class="eyebrow">{tr("OUR METHODOLOGY", "PHƯƠNG PHÁP LUẬN")}</div>'
                f'<div class="h2">{tr("How your ESG score is built.", "Điểm ESG của bạn được xây dựng như thế nào.")}</div>'
                f'<div class="lead" style="font-size:16px;max-width:760px;margin-bottom:26px">'
                + tr("The Data4ESGenius ESG assessment draws on the criteria structure of S&amp;P Global ESG Scores, the GRI and SASB reporting standards, "
                     "operational data supplied by Prime Group, and a weighting method based on materiality.",
                     "Đánh giá ESG của Data4ESGenius tham chiếu cấu trúc tiêu chí của S&amp;P Global ESG Scores, các chuẩn báo cáo GRI và SASB, "
                     "dữ liệu vận hành do Prime Group cung cấp và phương pháp gán trọng số theo mức độ trọng yếu.")
                + '</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="medium")
    with a1.container(key="card_m1"):
        st.markdown(
            f'<div class="h3" style="font-size:21px;margin-top:0">{tr("1. Reference frameworks", "1. Khung đánh giá tham chiếu")}</div>'
            '<div class="muted">'
            + tr(f'<b {D}>S&amp;P Global ESG Scores.</b> The S&amp;P ESG Score measures how well a company performs on and manages its material ESG risks, '
                 'opportunities and impacts, compared with peers in the same industry. Its input is the annual Corporate Sustainability Assessment (CSA), '
                 'combining company disclosures, media and stakeholder analysis, and modeling where disclosure is missing. '
                 'S&amp;P applies double materiality: an issue is material when it significantly affects society or the environment and also affects '
                 'the company\'s value drivers.<br><br>'
                 f'<b {D}>GRI</b> (Global Reporting Initiative) provides indicator-level measurement and disclosure standards, such as GRI 303 (Water) and GRI 306 (Waste). '
                 f'<b {D}>SASB</b> adds an industry-specific, investor-focused view of materiality.',
                 f'<b {D}>S&amp;P Global ESG Scores.</b> Điểm ESG của S&amp;P đo hiệu quả và cách quản lý các rủi ro, cơ hội và tác động ESG trọng yếu của doanh nghiệp, '
                 'so với các doanh nghiệp cùng ngành. Nguồn dữ liệu là Corporate Sustainability Assessment (CSA) hằng năm, kết hợp công bố của doanh nghiệp, '
                 'phân tích truyền thông và các bên liên quan, và mô hình hóa khi thiếu dữ liệu. '
                 'S&amp;P áp dụng cách tiếp cận trọng yếu kép (double materiality): một vấn đề là trọng yếu khi vừa tác động đáng kể đến xã hội/môi trường, '
                 'vừa tác động đến giá trị doanh nghiệp.<br><br>'
                 f'<b {D}>GRI</b> (Global Reporting Initiative) cung cấp chuẩn đo lường và công bố theo từng chỉ số, ví dụ GRI 303 (Nước) và GRI 306 (Chất thải). '
                 f'<b {D}>SASB</b> bổ sung góc nhìn trọng yếu theo ngành đối với nhà đầu tư.')
            + '</div>', unsafe_allow_html=True)
    with a2.container(key="card_m2"):
        st.markdown(
            f'<div class="h3" style="font-size:21px;margin-top:0">{tr("2. Two ways to produce a score", "2. Hai cách tạo điểm")}</div>'
            '<div class="muted">'
            + tr(f'<b {D}>Community Project (survey).</b> A 15-question survey split evenly across Environmental, Social and Governance &amp; Economic. '
                 'Each question has three maturity levels (none / basic / implemented and tracked), worth 0, 50 and 100 points. '
                 'A dimension score is the average of its 5 questions; the overall score is the average of the three dimensions.<br><br>'
                 f'<b {D}>Enterprise (company data).</b> Uploaded environmental indicators are validated and reported with configured weights and data coverage. '
                 'When an absolute benchmark is not available, Environment results are calculated from prior-period comparison, clearly presented as part of the available data assessment; Social and Governance are not scored.',
                 f'<b {D}>Community Project (khảo sát).</b> 15 câu hỏi, chia đều cho ba nhóm Môi trường, Xã hội, Quản trị &amp; Kinh tế. '
                 'Mỗi câu có 3 mức trưởng thành (chưa có / cơ bản / triển khai và theo dõi) tương ứng 0, 50 và 100 điểm. '
                 'Điểm mỗi nhóm là trung bình của 5 câu; điểm tổng là trung bình ba nhóm.<br><br>'
                 f'<b {D}>Enterprise (dữ liệu doanh nghiệp).</b> Các chỉ số môi trường tải lên được kiểm tra và báo cáo cùng trọng số, độ bao phủ dữ liệu. '
                 'Chưa tính điểm Environment khi chưa có benchmark hoặc ngưỡng được phê duyệt; không chấm điểm Social và Governance.')
            + '</div>', unsafe_allow_html=True)

    st.write("")
    with st.container(key="card_m3"):
        st.markdown(
            f'<div class="h3" style="font-size:21px;margin-top:0">{tr("3. The 35-criteria structure", "3. Cấu trúc 35 tiêu chí")}</div>'
            '<div class="muted">'
            + tr("The criteria follow the topic list of S&amp;P Global ESG Scores and are grouped into the 15 survey questions.",
                 "Các tiêu chí được xây dựng theo danh mục chủ đề của S&amp;P Global ESG Scores và nhóm lại thành 15 câu hỏi khảo sát.")
            + '</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, key, label in zip(cols, CRITERIA_KEYS, dim_names()):
            items = sorted({t for v in CRITERIA[key].values() for t in v})
            col.markdown(f'<div style="margin-top:14px"><b {D}>{label} ({len(items)})</b>'
                         f'<div class="muted" style="margin-top:6px;font-size:13px">' + " · ".join(items) + '</div></div>',
                         unsafe_allow_html=True)

    st.write("")
    with st.container(key="card_m4"):
        th = [tr("Indicator", "Chỉ số"), tr("Parameter", "Thông số"), tr("Unit", "Đơn vị"),
              tr("Materiality", "Trọng yếu"), tr("Basis", "Cơ sở")]
        tonne = tr("Tonnes", "Tấn")
        st.markdown(
            f'<div class="h3" style="font-size:21px;margin-top:0">{tr("5. Materiality, weights and benchmarks", "5. Trọng yếu, trọng số và ngưỡng so sánh")}</div>'
            '<div class="muted">'
            + tr('Each indicator is given a materiality level (key risk focus) <span class="tag t-high">High</span> <span class="tag t-med">Medium</span> '
                 '<span class="tag t-low">Low</span>, together with the reason for choosing it, its academic basis (GRI, ISO), practical basis, calculation basis and '
                 'benchmark/score threshold. The % weight of each indicator is assigned from this materiality level. The table shows two example indicators.',
                 'Mỗi chỉ số được gán mức trọng yếu (rủi ro trọng điểm) <span class="tag t-high">High</span> <span class="tag t-med">Medium</span> '
                 '<span class="tag t-low">Low</span>, kèm lý do chọn chỉ số, cơ sở học thuật (GRI, ISO), cơ sở thực tế, cơ sở tính toán và benchmark/ngưỡng điểm. '
                 'Trọng số % của từng chỉ số được gán dựa trên mức trọng yếu này. Bảng dưới là hai chỉ số minh họa.')
            + '</div>'
            f'<table class="mt"><tr>' + "".join(f"<th>{h}</th>" for h in th) + '</tr>'
            f'<tr><td rowspan="4"><b>Water</b></td><td>{tr("Total water withdrawal", "Tổng lượng nước rút")}</td><td>m³</td><td><span class="tag t-high">High</span></td>'
            '<td rowspan="4">GRI 303, ISO 14046</td></tr>'
            f'<tr><td>{tr("Water-use intensity", "Cường độ sử dụng nước")}</td><td>{tr("m³ per tonne of product", "m³/tấn sản phẩm")}</td><td></td></tr>'
            f'<tr><td>{tr("Water recycling / reuse rate", "Tỷ lệ tái chế / tái sử dụng nước")}</td><td>%</td><td></td></tr>'
            '<tr><td>Water stress</td><td></td><td></td></tr>'
            f'<tr><td rowspan="6"><b>Waste &amp; Hazardous Materials</b></td><td>{tr("Total waste generated", "Tổng lượng chất thải phát sinh")}</td><td>{tonne}</td><td><span class="tag t-med">Medium</span></td><td rowspan="6">GRI 306</td></tr>'
            f'<tr><td>{tr("Hazardous waste", "Chất thải nguy hại")}</td><td>{tonne}</td><td><span class="tag t-high">High</span></td></tr>'
            f'<tr><td>{tr("Non-hazardous waste", "Chất thải thông thường")}</td><td>{tonne}</td><td><span class="tag t-low">Low</span></td></tr>'
            f'<tr><td>{tr("Recycling / reuse rate", "Tỷ lệ tái chế / tái sử dụng")}</td><td>%</td><td><span class="tag t-high">High</span></td></tr>'
            f'<tr><td>{tr("Waste incinerated", "Chất thải đốt")}</td><td>{tonne}</td><td><span class="tag t-med">Medium</span></td></tr>'
            f'<tr><td>{tr("Waste landfilled", "Chất thải chôn lấp")}</td><td>{tonne}</td><td><span class="tag t-med">Medium</span></td></tr>'
            '</table>'
            '<div class="muted" style="margin-top:16px">'
            + tr(f'<b {D}>Calculation example (Water).</b> Total water withdrawal = surface water + groundwater + municipal water = 73,088 + 0 + 15,499 = 88,587 m³. '
                 f'The WBCSD/CSI benchmark is 1–4 m³ per tonne of product; the actual data is 0.52 m³/tonne, rated <b {D}>Excellent</b>.',
                 f'<b {D}>Ví dụ tính toán (Water).</b> Tổng lượng nước rút = nước mặt + nước ngầm + nước máy = 73.088 + 0 + 15.499 = 88.587 m³. '
                 f'Benchmark WBCSD/CSI là 1–4 m³/tấn sản phẩm; dữ liệu thực tế 0,52 m³/tấn, xếp mức <b {D}>Xuất sắc</b>.')
            + '</div><div class="muted" style="margin-top:14px">'
            + tr(f'<b {D}>Greenhouse gas reference (Vietnam).</b> The pilot emission allowance approved by the Government for three sectors (cement, steel, thermal power) '
                 'is 243 million tonnes of CO₂ in 2025 and 268.4 million tonnes in 2026. Cement alone accounts for about 28.8% (2025) and 28.1% (2026). '
                 'In 2025, national cement output was estimated at about 96.36 million tonnes; the allocated allowance was 69.94 million tonnes of CO₂ equivalent, '
                 'while actual emissions were estimated at about 84 million tonnes of CO₂. These figures were provided by the team.',
                 f'<b {D}>Tham chiếu phát thải khí nhà kính (Việt Nam).</b> Tổng hạn ngạch thí điểm đã được Chính phủ phê duyệt cho ba ngành xi măng, sắt thép, nhiệt điện: '
                 '243 triệu tấn CO₂ (2025) và 268,4 triệu tấn CO₂ (2026). Riêng xi măng chiếm khoảng 28,8% (2025) và 28,1% (2026). '
                 'Năm 2025, sản lượng xi măng ước đạt khoảng 96,36 triệu tấn; hạn ngạch được giao là 69,94 triệu tấn CO₂ tương đương, '
                 'trong khi phát thải thực tế ước khoảng 84 triệu tấn CO₂. Các số liệu này do nhóm cung cấp.')
            + '</div>', unsafe_allow_html=True)

        st.write("")
        with st.container(key="card_m_env_weights"):
           st.markdown(
              f'<div class="h3" style="font-size:21px;margin-top:0">{tr("5. Environment weights and scoring rules", "5. Trọng số Environment và quy tắc tính điểm")}</div>'
              '<div class="muted">'
              + tr("This Environment-only weighting is supplementary to the community survey. SASB Construction Materials is a proposed reference and must be confirmed against Prime Group’s actual industry before formal mapping.",
                  "Bộ trọng số Environment này bổ sung cho khảo sát dự án cộng đồng. SASB Construction Materials chỉ là khung tham chiếu dự kiến và cần xác minh ngành thực tế của Prime Group trước khi mapping chính thức.")
              + '</div>'
              f'<table class="mt"><tr><th>{tr("Environment group", "Nhóm Environment")}</th><th>{tr("Weight", "Trọng số")}</th><th>{tr("Materiality", "Mức trọng yếu")}</th></tr>'
              f'<tr><td>GHG Emissions</td><td>25%</td><td>{tr("Very High", "Rất cao")}</td></tr>'
              f'<tr><td>Energy Management</td><td>25%</td><td>{tr("Very High", "Rất cao")}</td></tr>'
              f'<tr><td>Water Management</td><td>15%</td><td>{tr("High", "Cao")}</td></tr>'
              f'<tr><td>Waste Management</td><td>15%</td><td>{tr("High", "Cao")}</td></tr>'
              f'<tr><td>Air Quality Emissions</td><td>20%</td><td>{tr("Very High", "Rất cao")}</td></tr>'
              f'<tr><td><b>{tr("Total", "Tổng")}</b></td><td><b>100%</b></td><td></td></tr></table>'
              '<div class="muted" style="margin-top:14px">'
              + tr('<b style="color:var(--dark)">Within-group indicator weights:</b><br>'
                  '<b>GHG:</b> Scope 1 45%, Scope 2 30%, Scope 3 10%, intensity 10%, reduction rate 5%.<br>'
                  '<b>Energy:</b> total consumption 15%, non-renewable consumption 25%, renewable ratio 20%, intensity 30%, reduction rate 10%.<br>'
                  '<b>Water:</b> withdrawal 15%, consumption 20%, intensity 30%, recycling/reuse rate 20%, water-stress exposure 15%.<br>'
                  '<b>Waste:</b> total generated 15%, hazardous 25%, non-hazardous 10%, recycling/recovery rate 30%, disposal/landfill rate 20%.<br>'
                  '<b>Air:</b> NOx 25%, SOx 25%, PM10/PM2.5 30%, other pollutants 10%, emission intensity 10%.',
                  '<b style="color:var(--dark)">Trọng số chỉ số trong từng nhóm:</b><br>'
                  '<b>GHG:</b> Scope 1 45%, Scope 2 30%, Scope 3 10%, cường độ 10%, tỷ lệ giảm 5%.<br>'
                  '<b>Năng lượng:</b> tổng tiêu thụ 15%, không tái tạo 25%, tỷ lệ tái tạo 20%, cường độ 30%, tỷ lệ giảm 10%.<br>'
                  '<b>Nước:</b> khai thác 15%, tiêu thụ 20%, cường độ 30%, tái chế/tái sử dụng 20%, căng thẳng nước 15%.<br>'
                  '<b>Chất thải:</b> tổng phát sinh 15%, nguy hại 25%, thông thường 10%, tái chế/thu hồi 30%, xử lý/chôn lấp 20%.<br>'
                  '<b>Không khí:</b> NOx 25%, SOx 25%, PM10/PM2.5 30%, chất ô nhiễm khác 10%, cường độ phát thải 10%.')
              + '</div>'
              '<div class="muted" style="margin-top:14px">'
              + tr('Normalize indicator scores to 0–100. With an approved benchmark, use the configured absolute normalization. Until a benchmark is approved, compare indicators with the previous reporting period. Calculate each group as the weighted sum and Environment as 25% GHG + 25% Energy + 15% Water + 15% Waste + 20% Air. Missing data is not zero, missing weights are disclosed and coverage is reported separately. GRI references: 305, 302, 303 and 306. SASB EM-CM mappings remain conditional on industry verification. PM2.5 is an internal indicator, not an asserted SASB requirement.',
                  'Chuẩn hóa điểm chỉ số về 0–100. Khi có benchmark được duyệt, dùng chuẩn hóa tuyệt đối đã cấu hình. Khi chưa có benchmark, có thể so sánh với kỳ trước để đánh giá xu hướng và mức thay đổi. Tính điểm nhóm bằng tổng có trọng số và Environment bằng 25% GHG + 25% Năng lượng + 15% Nước + 15% Chất thải + 20% Không khí. Dữ liệu thiếu không phải điểm 0, trọng số thiếu được công khai và độ bao phủ được báo cáo riêng. Tham chiếu GRI: 305, 302, 303 và 306. Mapping SASB EM-CM phụ thuộc xác minh ngành. PM2.5 là chỉ số nội bộ, không khẳng định là yêu cầu của SASB.')
              + '</div>', unsafe_allow_html=True)

    st.write("")
    with st.container(key="card_m5"):
        st.markdown(
            f'<div class="h3" style="font-size:21px;margin-top:0">{tr("6. References", "6. Tài liệu tham khảo")}</div>'
            '<div class="ref">S&amp;P Global. <i>ESG Scores &amp; Data</i>. spglobal.com/sustainable1/en/solutions/esg-scores-data</div>'
            '<div class="ref">Global Reporting Initiative (GRI). GRI 303: Water and Effluents; GRI 306: Waste.</div>'
            '<div class="ref">SASB Standards (Sustainability Accounting Standards Board).</div>'
            '<div class="ref">ISO 14046: Water footprint. WBCSD / Cement Sustainability Initiative (CSI).</div>'
            '<div class="ref">Dauerer, A. (2025). A systematic literature review of performance measurement systems and the '
            'integration of ESG factors. <i>Environmental and Sustainability Indicators</i>, 27, 100746.</div>'
            '<div class="ref">Gong, X., Tao, X., Zhang, M., Xu, Y., Kwok, H.H.L., Dai, J., &amp; Cheng, J.C.P. '
            'Secure environmental, social, and governance (ESG) data management for construction projects using blockchain. '
            '<i>Sustainable Cities and Society</i>.</div>'
            f'<div class="ref">{tr("Operational data source: Prime Group.", "Nguồn dữ liệu vận hành: Prime Group.")}</div>',
            unsafe_allow_html=True)
    st.markdown(f'<div class="small" style="margin-top:18px">{disclaimer()}</div>', unsafe_allow_html=True)


def steps_bar(active):
    labels = [tr("01  Project information", "01  Thông tin dự án"), tr("02  ESG survey", "02  Khảo sát ESG"),
              tr("03  Your results", "03  Kết quả của bạn")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="{"on" if i == active else ""}">{l}</div>' for i, l in enumerate(labels)) + '</div>',
        unsafe_allow_html=True)


def page_c_info():
    p = st.session_state.project
    st.markdown(f'<div class="eyebrow">{tr("COMMUNITY ASSESSMENT / STEP 1 OF 3", "ĐÁNH GIÁ CỘNG ĐỒNG / BƯỚC 1 TRÊN 3")}</div>',
                unsafe_allow_html=True)
    steps_bar(0)
    st.markdown(f'<div class="h2">{tr("Tell us about your project.", "Hãy cho chúng tôi biết về dự án của bạn.")}</div>'
                f'<div class="lead" style="font-size:16px;margin-bottom:24px">{tr("A few details help put your sustainability impact in context. No account needed.", "Một vài thông tin giúp đặt tác động bền vững của bạn vào đúng bối cảnh. Không cần tài khoản.")}</div>',
                unsafe_allow_html=True)
    lang = st.session_state.lang
    l, r = st.columns([1.65, 1], gap="medium")
    with l.container(key="card_6"):
        p["name"] = st.text_input(tr("Project Name", "Tên dự án"), p["name"], key=f"pn_{lang}",
                                  placeholder=tr("Green Neighborhood", "Khu phố xanh"))
        p["desc"] = st.text_area(tr("Project Description", "Mô tả dự án"), p["desc"], height=110, key=f"pd_{lang}",
                                 placeholder=tr("A community-led project that restores local green spaces and helps residents reduce waste through shared activities.",
                                                "Dự án do cộng đồng dẫn dắt, phục hồi không gian xanh địa phương và giúp người dân giảm rác thải thông qua các hoạt động chung."))
        c1, c2 = st.columns(2)
        p["location"] = c1.text_input(tr("Project Location", "Địa điểm dự án"), p["location"], key=f"pl_{lang}",
                                      placeholder=tr("Da Nang, Vietnam", "Đà Nẵng, Việt Nam"))
        p["duration"] = c2.text_input(tr("Project Duration", "Thời gian dự án"), p["duration"], key=f"pdu_{lang}",
                                      placeholder=tr("January – December 2026", "Tháng 1 – Tháng 12/2026"))
        cats = categories()
        p["category"] = st.selectbox(tr("Project Category", "Danh mục dự án"), range(len(cats)),
                                     index=p["category"], format_func=lambda i: cats[i], key=f"pc_{lang}")
        st.write("")
        b1, _, b2 = st.columns([1.5, 2.2, 1.4])
        b1.button(tr("Back to Home", "Về trang chủ"), on_click=go, args=("home",), use_container_width=True)
        b2.button(tr("Continue →", "Tiếp tục →"), type="primary", on_click=go, args=("c_survey",),
                  use_container_width=True)
    r.markdown(
        f'<div class="side"><h3>{tr("Small steps.<br>Meaningful insights.", "Những bước nhỏ.<br>Hiểu biết ý nghĩa.")}</h3>'
        f'<div class="st">01&nbsp; {tr("Share your project", "Chia sẻ dự án của bạn")}</div><div class="sd">{tr("Enter the basic details of your community initiative.", "Nhập thông tin cơ bản về sáng kiến cộng đồng của bạn.")}</div>'
        f'<div class="st">02&nbsp; {tr("Complete the survey", "Hoàn thành khảo sát")}</div><div class="sd">{tr("Answer questions across the three ESG dimensions.", "Trả lời các câu hỏi thuộc ba khía cạnh ESG.")}</div>'
        f'<div class="st">03&nbsp; {tr("Understand your impact", "Hiểu tác động của bạn")}</div><div class="sd">{tr("View your scores and download your assessment report.", "Xem điểm số và tải báo cáo đánh giá của bạn.")}</div>'
        f'<div class="sd" style="margin-top:22px;font-size:12px">{tr("Your answers are converted into scores automatically. You never enter an ESG score manually.", "Câu trả lời của bạn được tự động chuyển thành điểm số. Bạn không bao giờ phải nhập điểm ESG thủ công.")}</div></div>',
        unsafe_allow_html=True)


def page_c_survey():
    lang = st.session_state.lang
    questions = QUESTIONS_VI if lang == "vi" else QUESTIONS_EN
    secs = section_names()
    ans = st.session_state.answers
    sec = st.session_state.section
    name = st.session_state.project["name"] or tr("Green Neighborhood", "Khu phố xanh")
    st.markdown(f'<div class="eyebrow">{tr("COMMUNITY ASSESSMENT / STEP 2 OF 3", "ĐÁNH GIÁ CỘNG ĐỒNG / BƯỚC 2 TRÊN 3")}</div>',
                unsafe_allow_html=True)

    side, main = st.columns([1, 2.9], gap="large")
    with side:
        st.markdown(f'<div class="h2" style="font-size:30px;margin-bottom:4px">{tr("ESG Survey", "Khảo sát ESG")}</div>'
                    f'<div class="muted" style="margin-bottom:18px">{name}</div>', unsafe_allow_html=True)
        for i, s in enumerate(secs):
            done = all((i * 5 + j) in ans for j in range(5))
            if i == sec:
                status = tr("Current section", "Mục hiện tại")
            elif done and i < sec:
                status = tr("Completed", "Đã hoàn thành")
            else:
                status = tr("Up next", "Tiếp theo")
            st.markdown(f'<div class="sec-card {"cur" if i == sec else ""}"><div class="a">0{i+1}&nbsp; {s}</div>'
                        f'<div class="b">{status}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small">{tr("15 questions · 3 sections", "15 câu hỏi · 3 mục")}</div>'
                    f'<div class="small" style="margin-top:12px">{tr("Scores are calculated from survey answers, not entered manually.", "Điểm số được tính từ câu trả lời khảo sát, không nhập thủ công.")}</div>',
                    unsafe_allow_html=True)

    with main:
        n_ans = len(ans)
        with st.container(key="card_7"):
            st.markdown(f'<div style="display:flex;justify-content:space-between" class="muted"><span>{tr("Assessment progress", "Tiến độ đánh giá")}</span>'
                        f'<b style="color:#1F6F52;font-size:13px">{n_ans} / 15 {tr("answered", "đã trả lời")}</b></div>'
                        f'<div class="ptrack"><div class="pfill" style="width:{n_ans/15*100}%"></div></div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="h2" style="margin-top:22px">{secs[sec]}</div>'
                    f'<div class="muted" style="margin-bottom:18px">{tr("Choose the answer that best fits the project’s reality.", "Chọn một câu trả lời phù hợp nhất với thực tế dự án.")}</div>',
                    unsafe_allow_html=True)
        for qi in range(sec * 5, sec * 5 + 5):
            q, opts = questions[qi]
            with st.container(key=f"card_q{qi}"):
                choice = st.radio(f"{qi+1:02d}. {q}", opts, index=ans.get(qi), key=f"w_{qi}_{lang}")
                if choice is not None:
                    ans[qi] = opts.index(choice)

        st.write("")
        b1, _, b2 = st.columns([1.3, 2.7, 1.8])
        if sec > 0:
            b1.button(tr("← Previous", "← Trước"), on_click=lambda: st.session_state.update(section=sec - 1),
                      use_container_width=True)
        if sec < 2:
            b2.button(tr("Next →", "Tiếp →"), type="primary", on_click=lambda: st.session_state.update(section=sec + 1),
                      use_container_width=True)
        else:
            def submit():
                st.session_state.community_scores = community_scores()
                go("c_results")
            b2.button(tr("Submit Assessment", "Gửi đánh giá"), type="primary", on_click=submit,
                      disabled=len(ans) < 15, use_container_width=True)
        if sec == 2 and len(ans) < 15:
            st.caption(tr("Please answer all 15 questions to submit your assessment.",
                          "Vui lòng trả lời đủ 15 câu hỏi để gửi đánh giá."))
        st.markdown(f'<div class="hint">{tr("Your answers are converted into scores automatically. You never enter an ESG score manually.", "Câu trả lời của bạn được tự động chuyển thành điểm số. Bạn không bao giờ phải nhập điểm ESG thủ công.")}</div>',
                    unsafe_allow_html=True)


def page_c_results():
    s = st.session_state.community_scores or community_scores()
    name = st.session_state.project["name"] or tr("Green Neighborhood", "Khu phố xanh")
    sd = short_dims()
    best = sd[max(range(3), key=lambda i: s[DIM_KEYS[i]])]
    weak = sd[min(range(3), key=lambda i: s[DIM_KEYS[i]])]
    tell = tr(f"Your project shows a strong {best} contribution. More structured impact tracking and transparent reporting can strengthen its {weak} performance.",
              f"Dự án của bạn có đóng góp mạnh về {best}. Theo dõi tác động có cấu trúc hơn và báo cáo minh bạch có thể nâng cao kết quả về {weak}.")
    results_view(step_label=tr("COMMUNITY ASSESSMENT / STEP 3 OF 3", "ĐÁNH GIÁ CỘNG ĐỒNG / BƯỚC 3 TRÊN 3"),
                 title_sub=f"{name} · {tr('Community Project', 'Dự án cộng đồng')}",
                 note=tr("Scores are calculated automatically from your survey answers.",
                         "Điểm số được tính tự động từ câu trả lời khảo sát của bạn."),
                 s=s, tell=tell, hints=focus_hints(),
                 source=tr("Assessment source: ESG survey · 15 responses", "Nguồn đánh giá: Khảo sát ESG · 15 câu trả lời"),
                 back_label=tr("Back to Home", "Về trang chủ"), back_page="home", report_name="esg_community_report.html")


def page_ent_login():
    l, r = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    l.markdown(f'<div class="dark-panel"><div class="e">{tr("For enterprises", "Dành cho doanh nghiệp")}</div>'
               f'<div class="h">{tr("Your data.<br>A clearer ESG picture.", "Dữ liệu của bạn.<br>Bức tranh ESG rõ ràng hơn.")}</div>'
               f'<div class="p">{tr("Bring your company’s sustainability data together and turn it into meaningful insights.", "Tập hợp dữ liệu bền vững của công ty bạn và biến chúng thành những hiểu biết có ý nghĩa.")}</div>'
               f'<div class="s">01&nbsp; {tr("Upload company data", "Tải lên dữ liệu công ty")}</div><div class="d">{tr("Use CSV, XLS or XLSX files.", "Dùng tệp CSV, XLS hoặc XLSX.")}</div>'
               f'<div class="s">02&nbsp; {tr("Review Environment data", "Rà soát dữ liệu Environment")}</div><div class="d">{tr("Inspect coverage, monthly trends and potential anomalies.", "Xem độ bao phủ, xu hướng tháng và biến động cần rà soát.")}</div>'
               f'<div class="s">03&nbsp; {tr("Understand data limits", "Hiểu giới hạn dữ liệu")}</div><div class="d">{tr("Scores remain unavailable when benchmarks are not configured.", "Điểm chưa khả dụng khi chưa cấu hình benchmark.")}</div></div>',
               unsafe_allow_html=True)
    with r.container(key="card_9"):
        st.markdown(f'<div class="h2" style="font-size:30px;margin-top:8px">{tr("Welcome back.", "Chào mừng trở lại.")}</div>'
                    f'<div class="muted" style="margin-bottom:14px">{tr("Log in to your enterprise account to continue.", "Đăng nhập tài khoản doanh nghiệp để tiếp tục.")}</div>',
                    unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="sustainability@greenhorizon.vn")
        pw = st.text_input(tr("Password", "Mật khẩu"), type="password", placeholder="••••••••••••")
        st.button(tr("Login", "Đăng nhập"), type="primary", use_container_width=True, on_click=go, args=("ent_dash",))
        # TODO: replace on_click with real authentication (email, pw) before changing page.
        st.markdown(f'<div class="small" style="margin:8px 0 14px">{tr("Enterprise access is required to upload and assess company data.", "Cần quyền truy cập doanh nghiệp để tải lên và đánh giá dữ liệu công ty.")}</div>',
                    unsafe_allow_html=True)
        with st.container(key="guest_box"):
            st.markdown(f'<div style="background:var(--pale);border-radius:10px;padding:16px 16px 4px"><div style="color:var(--dark);font-size:14px;margin-bottom:10px">{tr("Assessing a community project?", "Bạn đang đánh giá một dự án cộng đồng?")}</div></div>',
                        unsafe_allow_html=True)
            st.button(tr("Continue as Guest →", "Tiếp tục với tư cách khách →"), on_click=go, args=("c_info",))


def enterprise_records():
    records = st.session_state.ent_records
    diagnostics = st.session_state.ent_diagnostics
    filename = st.session_state.ent_filename
    if records is not None:
        return records, diagnostics or {"duplicate_rows": 0, "unmapped_rows": 0}, filename or tr("Uploaded files", "Tệp đã tải lên")
    frames = load_extracted_files()
    records, diagnostics = normalize_frames(frames)
    return records, diagnostics, tr("Prime Group extracted CSV files", "Các tệp CSV đã trích xuất của Prime Group")


def render_source_records(records, indicator, flagged, widget_key):
    if flagged.empty:
        return
    periods = {
        (None if pd.isna(row.year) else int(row.year), row.month)
        for row in flagged[["year", "month"]].itertuples(index=False)
    }
    source_records = records[
        (records["indicator"] == indicator) & (records["data_type"] == "Monthly")
    ].copy()
    source_records = source_records.loc[
        source_records.apply(
            lambda row: (None if pd.isna(row["year"]) else int(row["year"]), row["month"]) in periods,
            axis=1,
        )
    ]
    if not source_records.empty:
        source_records["raw_record"] = source_records["raw_record"].map(str)
        with st.expander(tr("Original source records for flagged periods", "Bản ghi nguồn gốc của các kỳ được đánh dấu")):
            st.dataframe(source_records, hide_index=True, use_container_width=True)


def render_energy_insights(records, widget_key):
    monthly_records = records[records["data_type"] == "Monthly"]
    monthly_energy = monthly_records[monthly_records["category"] == "Energy Management"]
    st.subheader(tr("Monthly trend and anomaly review", "Xu hướng tháng và rà soát bất thường"))
    available_records = monthly_energy if widget_key == "dashboard" else monthly_records
    if available_records.empty:
        st.info(tr("No monthly energy observations are available for trend analysis.", "Không có quan sát năng lượng theo tháng để phân tích xu hướng."))
        return
    choices = sorted(available_records["indicator"].dropna().unique())
    selected_indicator = st.selectbox(tr("Monthly indicator", "Chỉ số theo tháng"), choices, key=f"{widget_key}_indicator")
    trend = monthly_trend(records, selected_indicator)
    if trend.empty:
        return
    trend = trend.copy()
    trend["period"] = trend.apply(
        lambda row: f"{int(row['year'])}-{row['month']}" if pd.notna(row["year"]) else row["month"], axis=1
    )
    trend["change_pct"] = trend["value"].pct_change() * 100
    trend.loc[trend["value"].shift(1) == 0, "change_pct"] = None
    st.line_chart(trend.set_index("period")["value"], y_label=trend["unit"].iloc[0])
    st.dataframe(trend[["year", "month", "value", "change_pct", "unit"]], hide_index=True, use_container_width=True)
    anomalies = z_score_anomalies(trend, threshold=2.5)
    if anomalies.empty:
        st.caption(tr("No unusual periods detected.", "Không phát hiện kỳ bất thường."))
    else:
        st.warning(tr("Some periods may need review.", "Một số kỳ có thể cần kiểm tra."))
        st.dataframe(anomalies[["year", "month", "value", "unit"]],
                     hide_index=True, use_container_width=True)
        render_source_records(records, selected_indicator, anomalies, f"{widget_key}_zscore")
    try:
        artifact = load_active_anomaly_model(selected_indicator)
    except Exception as error:
        artifact = None
        st.warning(tr(f"Could not load the active anomaly model: {error}", f"Không thể tải mô hình bất thường đang hoạt động: {error}"))
    if artifact is None:
        st.info(tr("No trained anomaly result is available for this indicator.",
                   "Chưa có kết quả bất thường đã huấn luyện cho chỉ số này."))
    else:
        trained_anomalies = infer_anomalies(trend, artifact)
        if trained_anomalies.empty:
            st.caption(tr("The trained model did not flag any periods for review.", "Mô hình đã huấn luyện không đánh dấu kỳ nào cần rà soát."))
        else:
            st.warning(tr("Please review the flagged periods.", "Vui lòng kiểm tra các kỳ được đánh dấu."))
            st.dataframe(trained_anomalies[["year", "month", "value", "unit"]],
                         hide_index=True, use_container_width=True)
            render_source_records(records, selected_indicator, trained_anomalies, f"{widget_key}_model")


def render_model_management(records):
    with st.expander(tr("ML Model Training and Management", "Huấn luyện và quản lý mô hình ML")):
        candidates = monthly_training_candidates(records)
        eligible = [item for item in candidates if item["eligible"]]
        st.caption(tr("Unsupervised Isolation Forest uses chronological train/validation/test partitions. No ESG or anomaly labels are fabricated.",
                      "Isolation Forest không giám sát dùng các tập train/validation/test theo thời gian. Không tự tạo nhãn ESG hay bất thường."))
        st.write(tr(f"Eligible indicators: {len(eligible)} · Minimum observations: 18", f"Chỉ số đủ điều kiện: {len(eligible)} · Số quan sát tối thiểu: 18"))
        if eligible:
            indicator_options = [item["indicator"] for item in eligible]
            indicator = st.selectbox(tr("Training indicator", "Chỉ số huấn luyện"), indicator_options, key="ml_training_indicator")
            selection = next(item for item in eligible if item["indicator"] == indicator)
            st.caption(tr(f"{selection['observations']} monthly observations · years {selection['first_year']}–{selection['last_year']}",
                          f"{selection['observations']} quan sát tháng · năm {selection['first_year']}–{selection['last_year']}"))
            contamination = st.selectbox(tr("Isolation Forest contamination", "Tỷ lệ contamination Isolation Forest"),
                                         [0.05, 0.1, 0.15, 0.2], index=1, key="ml_contamination")
            if st.button(tr("Train candidate model", "Huấn luyện mô hình ứng viên"), key="train_anomaly_model"):
                try:
                    with st.spinner(tr("Fitting on Prime Group records...", "Đang fit trên dữ liệu Prime Group...")):
                        persist_processed_records(records)
                        version = train_anomaly_model(records, indicator, contamination)
                    st.success(tr(f"Saved {version['version']} as {version['status']}. Previous active versions were preserved.",
                                  f"Đã lưu {version['version']} với trạng thái {version['status']}. Các phiên bản đang hoạt động trước đó được giữ nguyên."))
                    st.json(version["evaluation"])
                except Exception as error:
                    st.error(tr(f"Training failed: {error}", f"Huấn luyện thất bại: {error}"))
        else:
            st.warning(tr("No indicator currently meets the minimum training requirements. No model was trained.",
                          "Chưa có chỉ số nào đáp ứng điều kiện huấn luyện tối thiểu. Chưa huấn luyện mô hình."))

        models = list_anomaly_models()
        if models:
            st.subheader(tr("Saved model versions", "Các phiên bản mô hình đã lưu"))
            version_rows = [
                {
                    "Indicator": model_info["indicator"],
                    "Version": version_info["version"],
                    "Status": version_info["status"],
                    "Trained at": version_info["trained_at"],
                    "Records": version_info["number_of_records"],
                    "Train / validation / test": f"{version_info['train_records']} / {version_info['validation_records']} / {version_info['test_records']}",
                    "Validation alert rate": version_info["evaluation"]["validation_anomaly_rate"],
                    "Test alert rate": version_info["evaluation"]["test_anomaly_rate"],
                    "Active": version_info["version"] == model_info["active_version"],
                }
                for model_info in models
                for version_info in anomaly_model_versions(model_info["indicator"])
            ]
            st.dataframe(pd.DataFrame(version_rows), hide_index=True, use_container_width=True)
            versions = []
            for model_info in models:
                for version_info in anomaly_model_versions(model_info["indicator"]):
                    versions.append((model_info["indicator"], version_info["version"], version_info["status"]))
            selected_version = st.selectbox(
                tr("Version to activate", "Phiên bản cần kích hoạt"), versions,
                format_func=lambda item: f"{item[0]} · {item[1]} · {item[2]}", key="ml_promote_version",
            )
            if st.button(tr("Activate selected version", "Kích hoạt phiên bản đã chọn"), key="promote_anomaly_model"):
                try:
                    promoted = promote_anomaly_model(selected_version[0], selected_version[1])
                    st.success(tr(f"{promoted['version']} is now active for {selected_version[0]}.",
                                  f"{promoted['version']} hiện hoạt động cho {selected_version[0]}."))
                except Exception as error:
                    st.error(tr(f"Activation failed: {error}", f"Kích hoạt thất bại: {error}"))

        monthly_indicators = sorted(records.loc[records["data_type"] == "Monthly", "indicator"].dropna().unique())
        forecast_targets = [indicator for indicator in monthly_indicators if forecast_readiness(records, indicator)[0]]
        if forecast_targets:
            st.subheader(tr("SARIMA Forecast Training", "Huấn luyện dự báo SARIMA"))
            forecast_indicator = st.selectbox(tr("Forecast indicator", "Chỉ số dự báo"), forecast_targets,
                                             key="forecast_training_indicator")
            if st.button(tr("Train SARIMA model", "Huấn luyện mô hình SARIMA"), key="train_forecast_model"):
                try:
                    with st.spinner(tr("Training and evaluating SARIMA...", "Đang huấn luyện và đánh giá SARIMA...")):
                        forecast_metadata = train_forecast_model(records, forecast_indicator)
                    st.success(tr(f"Saved forecast {forecast_metadata['version']}.",
                                  f"Đã lưu mô hình dự báo {forecast_metadata['version']}."))
                except Exception as error:
                    st.error(tr(f"Forecast training failed: {error}", f"Huấn luyện dự báo thất bại: {error}"))
        else:
            st.info(tr("Forecasting is withheld: no indicator has 24 continuous monthly observations with a verified year. No forecast values are generated.",
                       "Chưa dự báo: không có chỉ số nào đạt 24 quan sát tháng liên tục có năm xác minh. Hệ thống không tạo giá trị dự báo."))

        forecast_models = list_forecast_models()
        if forecast_models:
            forecast_choices = [item["indicator"] for item in forecast_models]
            forecast_display_indicator = st.selectbox(tr("Forecast to display", "Dự báo cần hiển thị"), forecast_choices,
                                                       key="forecast_display_indicator")
            forecast_result = infer_forecast(forecast_display_indicator)
            if forecast_result is not None:
                forecast_display = forecast_result[["period", "forecast", "unit"]].copy()
                st.line_chart(forecast_display.set_index("period")[["forecast"]])
                st.dataframe(forecast_display, hide_index=True, use_container_width=True)


def page_ent_dash():
    records, diagnostics, filename = enterprise_records()
    assessment = calculate_environment_assessment(records)
    st.markdown(f'<div class="eyebrow">{tr("ENTERPRISE WORKSPACE", "KHÔNG GIAN DOANH NGHIỆP")}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
    c1.markdown(f'<div class="h2" style="margin-bottom:4px">{tr("Prime Group Environment assessment", "Đánh giá Environment của Prime Group")}</div>'
                f'<div class="lead" style="font-size:16px">{tr("Results and coverage are based on extracted source data. Social and Governance scores are not inferred.", "Kết quả và độ bao phủ dựa trên dữ liệu nguồn đã trích xuất. Không suy diễn điểm Social hoặc Governance.")}</div>',
                unsafe_allow_html=True)
    c2.button(tr("Upload Company Data ↑", "Tải lên dữ liệu công ty ↑"), type="primary", on_click=go,
              args=("ent_upload",), use_container_width=True)
    st.write("")
    columns = st.columns(4, gap="small")
    environment_score_display = (
        f"{assessment['score']:.1f}/100" if assessment["score"] is not None
        else tr("Not available", "Chưa thể tính")
    )
    columns[0].metric(tr("Environment score", "Điểm Environment"), environment_score_display)
    columns[1].metric(tr("Weighted data coverage", "Độ bao phủ dữ liệu có trọng số"), f"{assessment['coverage'] * 100:.1f}%")
    columns[2].metric(tr("Indicators with data", "Chỉ số có dữ liệu"), int(sum(item["available"] for group in assessment["groups"].values() for item in group["indicators"].values())))
    columns[3].metric(tr("Normalized records", "Bản ghi đã chuẩn hóa"), len(records))
    st.caption(f"{tr('Source', 'Nguồn')}: {filename} · {tr('Normalized records', 'Bản ghi đã chuẩn hóa')}: {len(records)}")
    st.info(tr("Results are calculated from available environmental data and the configured Environment weights.",
               "Kết quả được tính từ dữ liệu môi trường có sẵn và trọng số Environment đã cấu hình."))
    group_rows = [{
        "Environment group": group,
        "Weight": f"{details['weight']:.0%}",
        "Data coverage": f"{details['coverage']:.0%}",
        "Score": f"{details['score']:.1f}" if details["score"] is not None else tr("Not available", "Chưa thể tính"),
    } for group, details in assessment["groups"].items()]
    st.dataframe(pd.DataFrame(group_rows), hide_index=True, use_container_width=True)
    render_energy_insights(records, "dashboard")
    st.button(tr("View Environment analysis →", "Xem phân tích Environment →"), type="primary", on_click=go,
              args=("ent_analysis",))
    if diagnostics.get("duplicate_rows"):
        st.warning(tr(f"{diagnostics['duplicate_rows']} exact duplicate source rows were found. Review the source records before reporting.",
                      f"Phát hiện {diagnostics['duplicate_rows']} dòng nguồn trùng khớp hoàn toàn. Hãy rà soát trước khi báo cáo."))
    report = pd.DataFrame(group_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(tr("Export Environment coverage report", "Xuất báo cáo độ bao phủ Environment"),
                       data=report, file_name="prime_group_environment_coverage.csv", mime="text/csv")


def page_ent_upload():
    st.markdown(f'<div class="eyebrow">{tr("ENTERPRISE ASSESSMENT / UPLOAD", "ĐÁNH GIÁ DOANH NGHIỆP / TẢI LÊN")}</div>'
                f'<div class="h2">{tr("Start with your company data.", "Bắt đầu với dữ liệu công ty của bạn.")}</div>'
                f'<div class="lead" style="font-size:16px;margin-bottom:24px">{tr("Upload extracted environmental CSV files or a workbook. Only supported Environment data is assessed.", "Tải lên các CSV môi trường đã trích xuất hoặc workbook. Chỉ dữ liệu Environment được hỗ trợ mới được đánh giá.")}</div>',
                unsafe_allow_html=True)
    l, r = st.columns([2.1, 1], gap="medium")
    with l.container(key="card_11"):
        st.markdown(f'<div class="h3" style="font-size:21px;margin:6px 0 14px">{tr("Upload Company Data", "Tải lên dữ liệu công ty")}</div>',
                    unsafe_allow_html=True)
        files = st.file_uploader(tr("Drag and drop your files here", "Kéo và thả các tệp của bạn vào đây"),
                     type=["csv", "xls", "xlsx"], accept_multiple_files=True, label_visibility="collapsed")
        st.markdown(f'<div class="small" style="text-align:center;margin-top:6px">{tr("Supported formats: CSV, XLS, XLSX", "Định dạng hỗ trợ: CSV, XLS, XLSX")}</div>',
                    unsafe_allow_html=True)
        if files:
            st.markdown(f'<div class="filechip"><div><b>{len(files)} {tr("file(s) selected", "tệp đã chọn")}</b></div>'
                        f'<span class="r">{tr("Ready", "Sẵn sàng")}</span></div>', unsafe_allow_html=True)
            try:
                frames = load_uploaded_files(files)
                source_names = list(frames)
                selected_source = st.selectbox(tr("Preview source", "Xem trước nguồn dữ liệu"), source_names)
                st.dataframe(frames[selected_source].head(20), use_container_width=True)
            except Exception as error:
                st.error(tr(f"Could not read uploaded data: {error}", f"Không thể đọc dữ liệu đã tải lên: {error}"))
                frames = {}
        else:
            frames = {}
        st.write("")
        b1, _, b2 = st.columns([1.7, 1.6, 1.6])
        b1.button(tr("Back to Dashboard", "Về bảng điều khiển"), on_click=go, args=("ent_dash",), use_container_width=True)
        if b2.button(tr("Analyze Data →", "Phân tích dữ liệu →"), type="primary", disabled=not frames,
                      use_container_width=True):
            normalized, diagnostics = normalize_frames(frames)
            if normalized.empty:
                st.error(tr("No supported Environment indicators were found in the selected files.",
                            "Không tìm thấy chỉ số Environment được hỗ trợ trong các tệp đã chọn."))
            else:
                st.session_state.ent_records = normalized
                st.session_state.ent_diagnostics = diagnostics
                st.session_state.ent_filename = ", ".join(file.name for file in files)
                try:
                    persist_processed_records(normalized)
                except OSError as error:
                    st.warning(tr(f"Could not persist the processed copy: {error}", f"Không thể lưu bản dữ liệu đã xử lý: {error}"))
                go("ent_analysis")
                st.rerun()
    r.markdown(f'<div class="side" style="background:#fff"><h3 style="font-size:22px">{tr("Before you upload", "Trước khi tải lên")}</h3>'
               f'<div class="st">{tr("Use a supported format", "Dùng định dạng được hỗ trợ")}</div><div class="sd">{tr("Choose a CSV, XLS or XLSX spreadsheet.", "Chọn bảng tính CSV, XLS hoặc XLSX.")}</div>'
               f'<div class="st">{tr("Include available Environment data", "Bao gồm dữ liệu Environment hiện có")}</div><div class="sd">{tr("Upload only the environmental indicators available in your source files.", "Chỉ tải lên các chỉ số môi trường có trong tệp nguồn.")}</div>'
               f'<div class="st">{tr("Check the data", "Kiểm tra dữ liệu")}</div><div class="sd">{tr("Make sure column names are clear and values are consistent.", "Đảm bảo tên cột rõ ràng và giá trị nhất quán.")}</div>'
               f'<div class="hint">{tr("Missing periods or indicators receive no score.", "Chỉ số hoặc kỳ thiếu không được gán điểm.")}</div></div>',
               unsafe_allow_html=True)


def page_ent_analysis():
    records, diagnostics, filename = enterprise_records()
    assessment = calculate_environment_assessment(records)
    st.markdown(f'<div class="eyebrow">{tr("ENTERPRISE ASSESSMENT / ENVIRONMENT ANALYSIS", "ĐÁNH GIÁ DOANH NGHIỆP / PHÂN TÍCH ENVIRONMENT")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="h2">{tr("Environment data assessment", "Đánh giá dữ liệu Environment")}</div>'
                f'<div class="lead" style="font-size:16px;margin-bottom:20px">{filename}</div>', unsafe_allow_html=True)
    if records.empty:
        st.warning(tr("No extracted or uploaded environmental records are available.", "Chưa có dữ liệu môi trường được trích xuất hoặc tải lên."))
        st.button(tr("Upload data", "Tải dữ liệu lên"), on_click=go, args=("ent_upload",))
        return

    metric_cols = st.columns(3)
    environment_score_display = (
        f"{assessment['score']:.1f}/100" if assessment["score"] is not None
        else tr("Not available", "Chưa thể tính")
    )
    metric_cols[0].metric(tr("Environment score", "Điểm Environment"), environment_score_display)
    metric_cols[1].metric(tr("Weighted data coverage", "Độ bao phủ dữ liệu có trọng số"), f"{assessment['coverage'] * 100:.1f}%")
    metric_cols[2].metric(tr("Normalized records", "Bản ghi đã chuẩn hóa"), len(records))
    st.info(tr("Results are calculated from available environmental data and the configured Environment weights.",
               "Kết quả được tính từ dữ liệu môi trường có sẵn và trọng số Environment đã cấu hình."))

    st.subheader(tr("Environment groups and data coverage", "Nhóm Environment và độ bao phủ dữ liệu"))
    group_rows = []
    indicator_rows = []
    for group, details in assessment["groups"].items():
        group_rows.append({
            "Group": group,
            "Weight": details["weight"],
            "Data coverage": details["coverage"],
            "Score": round(details["score"], 1) if details["score"] is not None else None,
            "Status": tr("Available" if details["score"] is not None else "Unavailable",
                          "Có dữ liệu" if details["score"] is not None else "Chưa đủ dữ liệu"),
        })
        for name, indicator in details["indicators"].items():
            indicator_rows.append({
                "Group": group,
                "Indicator": name,
                "Within-group weight": indicator["weight"],
                "Available": tr("Yes", "Có") if indicator["available"] else tr("No", "Không"),
                "Score": indicator["score"],
                "Actual Value": indicator["actual_value"],
                "Previous Value": indicator["previous_value"],
                "Change %": round(indicator["change_pct"], 2) if indicator["change_pct"] is not None else None,
                "Status": tr("Available" if indicator["score"] is not None else ("No data" if not indicator["available"] else "Insufficient period data"),
                              "Có dữ liệu" if indicator["score"] is not None else ("Thiếu dữ liệu" if not indicator["available"] else "Chưa đủ kỳ dữ liệu")),
            })
    group_table = pd.DataFrame(group_rows)
    group_table["Weight"] = group_table["Weight"].map(lambda value: f"{value:.0%}")
    group_table["Data coverage"] = group_table["Data coverage"].map(lambda value: f"{value:.0%}")
    st.dataframe(group_table, hide_index=True, use_container_width=True)
    with st.expander(tr("Indicator-level assessment details", "Chi tiết đánh giá từng chỉ số")):
        st.dataframe(pd.DataFrame(indicator_rows), hide_index=True, use_container_width=True)

    render_energy_insights(records, "assessment")
    render_model_management(records)

    st.caption(tr(f"Source files: {diagnostics.get('source_files', 0)} · Exact duplicate rows: {diagnostics.get('duplicate_rows', 0)} · Actual zero values: {diagnostics.get('zero_values', 0)} · Negative values: {diagnostics.get('negative_values', 0)} · Unmapped: {diagnostics.get('unmapped_rows', 0)}",
                  f"Tệp nguồn: {diagnostics.get('source_files', 0)} · Dòng trùng khớp: {diagnostics.get('duplicate_rows', 0)} · Giá trị 0 thực tế: {diagnostics.get('zero_values', 0)} · Giá trị âm: {diagnostics.get('negative_values', 0)} · Chưa ánh xạ: {diagnostics.get('unmapped_rows', 0)}"))
    with st.expander(tr("Normalized source data", "Dữ liệu nguồn đã chuẩn hóa")):
        period_types = [tr("All periods", "Tất cả kỳ"), "Monthly", "Quarterly", "YTD"]
        selected_period_type = st.selectbox(tr("Reporting data type", "Loại kỳ báo cáo"), period_types)
        preview = records if selected_period_type == period_types[0] else records[records["data_type"] == selected_period_type]
        st.dataframe(preview.drop(columns=["raw_record"]), hide_index=True, use_container_width=True)


PAGES = {
    "home": page_home,
    "methodology": page_methodology,
    "c_info": page_c_info,
    "c_survey": page_c_survey,
    "c_results": page_c_results,
    "ent_login": page_ent_login,
    "ent_dash": page_ent_dash,
    "ent_upload": page_ent_upload,
    "ent_analysis": page_ent_analysis,
    "ent_results": page_ent_analysis,
}
header()
PAGES.get(st.session_state.page, page_home)()
footer()
