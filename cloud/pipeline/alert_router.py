"""
alert_router.py — PuppyPi Alert Router (Local / Laptop-as-Cloud Version)
=========================================================================
Sends alerts via Gmail SMTP (email) instead of AWS SES/SNS.
No AWS credentials or internet-facing services required.
Works entirely over the PuppyPi hotspot local network.

Setup (one-time):
  1. Go to your Google Account → Security → 2-Step Verification → App Passwords
  2. Create an App Password for "Mail"
  3. Fill in GMAIL_SENDER, GMAIL_APP_PASSWORD, and EMAIL_RECIPIENTS below

Alert severity routing:
  CRITICAL  → Email immediately
  HIGH      → Email immediately
  MEDIUM    → Email only
  LOW       → Log only (no email, avoids noise)
"""

import logging
import smtplib
import hashlib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

log = logging.getLogger("alert_router")

# ── Config — fill these in ────────────────────────────────────────────────────
GMAIL_SENDER       = "your.gmail@gmail.com"       # sender Gmail address
GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"        # Gmail App Password (not your login password)

EMAIL_RECIPIENTS   = [
    "member1@example.com",
    "member2@example.com",
    "member3@example.com",
    "member4@example.com",
    "member5@example.com"
]

# Alert cooldown — won't re-send same alert type within this window (seconds)
COOLDOWN_SECONDS = {
    "INTRUDER": 300,    # 5 min — avoid spam if person stays in zone
    "HAZARD":   120,    # 2 min — gas alerts are more urgent
    "SYSTEM":   1800,   # 30 min — battery / system warnings
}

# ── Rate Limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Prevents alert storms by enforcing a cooldown per alert type + dedup key.
    In-memory only — resets when the process restarts.
    """
    _last_sent: dict = {}

    @classmethod
    def should_send(cls, alert_type: str, dedup_key: str) -> bool:
        key      = hashlib.md5(f"{alert_type}:{dedup_key}".encode()).hexdigest()
        cooldown = COOLDOWN_SECONDS.get(alert_type, 300)
        last     = cls._last_sent.get(key, 0)

        if time.time() - last < cooldown:
            remaining = int(cooldown - (time.time() - last))
            log.info(f"Rate limited: {alert_type}/{dedup_key} — cooldown {remaining}s remaining")
            return False

        cls._last_sent[key] = time.time()
        return True

# ── Email Builder ─────────────────────────────────────────────────────────────
def _build_html_email(alert_type: str, subject: str, message: str, severity: str) -> str:
    """Build a simple HTML email body."""
    color_map = {
        "CRITICAL": "#DC2626",
        "HIGH":     "#EA580C",
        "MEDIUM":   "#D97706",
        "LOW":      "#2563EB",
    }
    icon_map = {
        "INTRUDER": "🚨",
        "HAZARD":   "☣️",
        "SYSTEM":   "⚙️",
    }
    badge_color = color_map.get(severity, "#6B7280")
    icon        = icon_map.get(alert_type, "⚠️")
    timestamp   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message_html = "".join(
        f"<p style='margin:4px 0;color:#374151;font-size:15px;font-family:monospace'>{line}</p>"
        for line in message.strip().split("\n") if line.strip()
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#F3F4F6;font-family:monospace;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding:28px 16px;">
          <table width="560" cellpadding="0" cellspacing="0"
            style="background:#fff;border-radius:10px;overflow:hidden;
                   box-shadow:0 4px 20px rgba(0,0,0,0.08);">

            <tr><td style="background:#111827;padding:22px 28px;">
              <p style="margin:0;font-size:10px;letter-spacing:4px;color:#6B7280;">
                PUPPYPI SECURITY SYSTEM
              </p>
              <h1 style="margin:8px 0 0;font-size:20px;color:#F9FAFB;font-weight:900;">
                {icon} {subject}
              </h1>
            </td></tr>

            <tr><td style="padding:14px 28px 0;">
              <span style="display:inline-block;background:{badge_color};color:#fff;
                           font-size:10px;font-weight:700;letter-spacing:2px;
                           padding:3px 10px;border-radius:4px;">
                {severity} SEVERITY
              </span>
            </td></tr>

            <tr><td style="padding:16px 28px;">
              <div style="background:#F9FAFB;border:1px solid #E5E7EB;
                          border-radius:8px;padding:14px 18px;">
                {message_html}
              </div>
            </td></tr>

            <tr><td style="padding:12px 28px 20px;border-top:1px solid #F3F4F6;">
              <p style="margin:0;font-size:10px;color:#9CA3AF;">
                Sent at {timestamp} · PuppyPi Edge Security System (INF2009)
              </p>
            </td></tr>

          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

# ── Email Sender ──────────────────────────────────────────────────────────────
def _send_email(subject: str, message: str, alert_type: str, severity: str):
    """
    Send alert email via Gmail SMTP SSL.
    Uses your Gmail App Password — does NOT need AWS or internet beyond Gmail.
    """
    if not EMAIL_RECIPIENTS:
        log.warning("No email recipients configured — skipping email")
        return

    if not GMAIL_APP_PASSWORD or GMAIL_APP_PASSWORD == "xxxx xxxx xxxx xxxx":
        log.warning("Gmail App Password not configured — logging alert instead")
        log.warning(f"[ALERT] {subject}\n{message}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = ", ".join(EMAIL_RECIPIENTS)

    # Attach both plain text and HTML versions
    msg.attach(MIMEText(message, "plain"))
    msg.attach(MIMEText(_build_html_email(alert_type, subject, message, severity), "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
        log.info(f"Email sent to {len(EMAIL_RECIPIENTS)} recipient(s): {subject}")

    except smtplib.SMTPAuthenticationError:
        log.error("Gmail authentication failed — check GMAIL_APP_PASSWORD")
        log.error("Make sure you're using an App Password, not your Gmail login password")
        log.error("Generate one at: myaccount.google.com → Security → App Passwords")

    except smtplib.SMTPException as e:
        log.error(f"SMTP error sending email: {e}")

    except Exception as e:
        log.error(f"Unexpected error sending email: {e}")

# ── Main Alert Router ─────────────────────────────────────────────────────────
class AlertRouter:
    """
    Central alert dispatcher called by cloud_subscriber.py.

    Usage:
        router = AlertRouter()
        router.send_alert(
            alert_type = "INTRUDER",
            subject    = "Intruder Detected",
            message    = "Zone: Main\\nConfidence: 94%\\nTime: 14:32",
            severity   = "HIGH",
            dedup_key  = "event-abc123",
        )

    Severity routing:
        CRITICAL / HIGH  → Email + console log
        MEDIUM           → Email + console log
        LOW              → Console log only
    """

    def send_alert(
        self,
        alert_type: str,
        subject:    str,
        message:    str,
        severity:   str,
        dedup_key:  Optional[str] = None,
    ):
        key = dedup_key or alert_type

        # Rate limit check — silently skip if within cooldown
        if not RateLimiter.should_send(alert_type, key):
            return

        # Always log to console regardless of severity
        log.warning(f"[{severity}] [{alert_type}] {subject}")
        for line in message.strip().split("\n"):
            log.warning(f"  {line}")

        # Email for CRITICAL / HIGH / MEDIUM
        if severity in ("CRITICAL", "HIGH", "MEDIUM"):
            _send_email(subject, message, alert_type, severity)

        # LOW — console log only, no email
        else:
            log.info(f"LOW severity alert logged only (no email): {subject}")

# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick test — run this directly to verify your Gmail setup works:
        python3 cloud/pipeline/alert_router.py
    Check your inbox after running.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    router = AlertRouter()

    print("Sending test alerts...")

    # Test HIGH — should send email
    router.send_alert(
        alert_type = "INTRUDER",
        subject    = "TEST — Intruder Detected",
        message    = (
            "Zone: Main Entrance\n"
            "Confidence: 94.2%\n"
            "Time: 2025-01-01T12:00:00\n"
            "Gas PPM: 145.0\n"
            "Temp: 28.5°C"
        ),
        severity   = "HIGH",
        dedup_key  = "test-intruder",
    )

    # Test CRITICAL — should send email
    router.send_alert(
        alert_type = "HAZARD",
        subject    = "TEST — Critical Gas Level",
        message    = (
            "Gas level: 550.0 PPM (CRITICAL)\n"
            "Temperature: 42.0°C\n"
            "Time: 2025-01-01T12:00:05"
        ),
        severity   = "CRITICAL",
        dedup_key  = "test-hazard",
    )

    # Test LOW — should only log, no email
    router.send_alert(
        alert_type = "SYSTEM",
        subject    = "TEST — Low Battery",
        message    = "Battery at 12%. Please recharge.",
        severity   = "LOW",
        dedup_key  = "test-battery",
    )

    print("\nDone. Check your inbox and console output above.")
    print("If no email arrived, check GMAIL_SENDER and GMAIL_APP_PASSWORD config.")