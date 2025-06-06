import smtplib
from email.message import EmailMessage
import os

# Load credentials from environment variables
EMAIL_ADDRESS = os.environ["GMAIL_USER"]
EMAIL_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
COMPANY_EMAIL = os.environ["VISA_MLP_EMAIL"]

# Compose email
msg = EmailMessage()
msg["Subject"] = "[Automated Email] Current Background Jobs Done on B200 ✅"
msg["From"] = EMAIL_ADDRESS
msg["To"] = EMAIL_ADDRESS
msg["CC"] = COMPANY_EMAIL
msg.set_content("Please check the logs and run the next test.")

# Send email via Gmail SMTP
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("✅ Email sent successfully.")
