import csv
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import qrcode

# === CONFIGURATION ===
CHALLAN_DIR = "echallans"
QR_DIR = "qr_codes"
CSV_FILE = "violations_log.csv"
VEHICLE_FILE = "vehicles.csv"

EMAIL_CREDENTIALS = {
    "sender_email": "yourproject@gmail.com",   # <-- replace with your email
    "app_password": "your_app_password_here"   # <-- replace with your app password
}

os.makedirs(CHALLAN_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

FINES = {
    "Red Light Violation": 1000,
    "Speed Limit Violation": 1500,
    "Unknown": 500
}

# === Read registered vehicle details ===
vehicle_emails = {}
if os.path.exists(VEHICLE_FILE):
    with open(VEHICLE_FILE, "r") as vf:
        reader = csv.DictReader(vf)
        for row in reader:
            plate = row["LicensePlate"].strip()
            vehicle_emails[plate] = {
                "name": row["OwnerName"].strip(),
                "email": row["Email"].strip()
            }
else:
    print("⚠️ vehicles.csv not found. Proceeding without email lookup.")

# === Read violations ===
if not os.path.exists(CSV_FILE):
    print("❌ No violations_log.csv found.")
    exit()

with open(CSV_FILE, "r") as vf:
    reader = csv.DictReader(vf)
    violations = list(reader)

if not violations:
    print("⚠️ No violations recorded.")
    exit()

def send_email(to_email, subject, body, attachment_path=None):
    msg = EmailMessage()
    msg["From"] = EMAIL_CREDENTIALS["sender_email"]
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=os.path.basename(attachment_path))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_CREDENTIALS["sender_email"], EMAIL_CREDENTIALS["app_password"])
            smtp.send_message(msg)
        print(f"📧 Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# === Generate challans and email ===
for i, v in enumerate(violations, start=1):
    timestamp = v.get("Timestamp", "Unknown")
    violation_id = v.get("ViolationID", str(i))
    plate = v.get("LicensePlate", "Unknown")
    image = v.get("Image", "")
    video = v.get("Video", "")

    violation_type = "Speed Limit Violation" if "speed" in video.lower() else \
                     "Red Light Violation" if "red" in video.lower() else "Unknown"

    fine_amount = FINES.get(violation_type, 500)
    challan_filename = os.path.join(CHALLAN_DIR, f"eChallan_{violation_id}.pdf")

    # === Generate QR ===
    payment_url = f"https://smarttrafficpay.gov/pay?challan_id={violation_id}&amount={fine_amount}"
    qr = qrcode.make(payment_url)
    qr_filename = os.path.join(QR_DIR, f"qr_{violation_id}.png")
    qr.save(qr_filename)

    # === Generate PDF ===
    c = canvas.Canvas(challan_filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(180, height - 80, "e-Challan Notification")
    c.setStrokeColor(colors.gray)
    c.line(50, height - 90, width - 50, height - 90)

    c.setFont("Helvetica", 12)
    y = height - 130
    c.drawString(60, y, f"Challan No: {violation_id}")
    c.drawString(60, y - 20, f"Vehicle No: {plate}")
    c.drawString(60, y - 40, f"Violation: {violation_type}")
    c.drawString(60, y - 60, f"Timestamp: {timestamp}")
    c.drawString(60, y - 80, f"Fine Amount: ₹{fine_amount}")

    if os.path.exists(image):
        c.drawImage(image, 60, y - 300, width=200, preserveAspectRatio=True, mask='auto')

    if os.path.exists(qr_filename):
        c.drawImage(qr_filename, width - 250, y - 250, width=150, preserveAspectRatio=True, mask='auto')
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(width - 250, y - 260, "Scan to pay online")

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(60, 80, "Please pay within 7 days to avoid penalty.")
    c.drawString(60, 60, "Issued by: Smart Traffic Violation Detection System")
    c.save()

    print(f"✅ e-Challan generated: {challan_filename}")

    # === Send email ===
    vehicle_info = vehicle_emails.get(plate)
    if vehicle_info:
        owner = vehicle_info["name"]
        email = vehicle_info["email"]
        body = (
            f"Dear {owner},\n\n"
            f"Your vehicle ({plate}) has been recorded for {violation_type} on {timestamp}.\n"
            f"Fine Amount: ₹{fine_amount}\n"
            f"You can scan the QR code in the attached challan PDF or visit:\n{payment_url}\n\n"
            f"Regards,\nSmart Traffic Violation Detection System"
        )
        send_email(email, "Traffic Violation e-Challan", body, challan_filename)
    else:
        print(f"⚠️ No registered email found for {plate}.")
