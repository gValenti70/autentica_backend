import os

# ==============================
# AZURE OPENAI
# ==============================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://autenticagpt.openai.azure.com/"
)

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")

AZURE_OPENAI_VERSION = os.getenv(
    "AZURE_OPENAI_VERSION",
    "2024-12-01-preview"
)

DEPLOYMENT_GPT = os.getenv("DEPLOYMENT_GPT", "gpt-5.1-chat")


# ==============================
# APP CONFIG
# ==============================

MAX_FOTO = int(os.getenv("MAX_FOTO", "5"))


# ==============================
# SMTP
# ==============================

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM")


# ==============================
# MONGO COLLECTIONS
# ==============================

analisi_col = "aut_analisi"
foto_col = "aut_analisi_foto"
prompts_col = "aut_prompts"
prompt_versions_col = "aut_prompt_versions"
users_col = "aut_users"
vademecum_col = "aut_vademecum"
login_log_col = "aut_login_log"
