import os
import smtplib
from email.mime.text import MIMEText


def invia_mail_perizia(email, html):

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    mail_from = os.getenv("MAIL_FROM")

    if not smtp_host:
        raise RuntimeError("SMTP_HOST non configurato")

    msg = MIMEText(html, "html")

    msg["Subject"] = "Richiesta perizia - Autentica"
    msg["From"] = mail_from
    msg["To"] = email

    with smtplib.SMTP(smtp_host, smtp_port) as s:

        s.starttls()

        if smtp_user and smtp_pass:
            s.login(smtp_user, smtp_pass)

        s.send_message(msg)
