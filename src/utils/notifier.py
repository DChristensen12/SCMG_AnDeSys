import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_spill_alert(spill_count, locations_affected):
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")
    
    if not all([sender, password, receiver]):
        print("Alert skipped: Email credentials missing in .env")
        return

    subject = f" ALERT: {spill_count} Potential Spill(s) Detected in Strawberry Creek"
    
    body = f"""
    The SCMG Anomaly Detection System has identified potential spills.
    
    Count: {spill_count}
    Affected Locations: {', '.join(locations_affected)}
    Timestamp: {os.popen('date').read()}
    
    Please check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Spill alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")