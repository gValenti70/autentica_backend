import os

# ==============================
# OPENAI
# ==============================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://autenticagpt.openai.azure.com/"
)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION",
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
