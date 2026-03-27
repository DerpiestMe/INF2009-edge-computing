"""
alert_router.py — PuppyPi Alert Routing (Email + SMS)
======================================================
Handles all outbound notifications for intrusion and hazard events.
Uses AWS SES (email) and AWS SNS (SMS) to route alerts.

Alert severity levels and their routing:
  CRITICAL  → SMS + Email immediately
  HIGH      → SMS + Email immediately
  MEDIUM    → Email only
  LOW       → Email only (batched, non-urgent)

Includes:
  - Rate limiting (prevents alert storms)
  - Alert deduplication (won't re-alert same event within cooldown)
  - HTML email templates with event details
  - Configurable recipient list from AWS SSM Parameter Store

AWS Services used:
  - SES : Transactional email (HTML templates)
  - SNS : SMS to phone numbers
  - SSM : Secure storage of recipient config
"""

import json
import time
import logging
import hashlib
import boto3
from datetime import datetime, timezone
from typing import Optional
from botocore.exceptions import ClientError

log = logging.getLogger("alert_router")

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION        = "ap-southeast-1"
SES_SENDER_EMAIL  = "alerts@your-domain.com"   # Must be SES-verified
SSM_PARAM_PATH    = "/puppypi/alerts"           # SSM path for config

# Alert cooldown — won't send duplicate alerts for the same type within this window
COOLDOWN_SECONDS  = {
    "INTRUDER": 300,   # 5 min cooldown per zone
    "HAZARD":   120,   # 2 min cooldown (hazards are more urgent)
    "SYSTEM":   1800,  # 30 min cooldown for system alerts (battery, etc.)
}

# ── AWS Clients ───────────────────────────────────────────────────────────────
ses = boto3.client("ses",  region_name=AWS_REGION)
sns = boto3.client("sns",  region_name=AWS_REGION)
ssm = boto3.client("ssm",  region_name=AWS_REGION)

# ── Recipient Config ──────────────────────────────────────────────────────────
class RecipientConfig:
    """
    Loads alert recipient list from SSM Parameter Store.
    Cached for 5 minutes to avoid excessive SSM calls.
    
    SSM parameters expected:
      /puppypi/alerts/email_recipients  → JSON list of email addresses
      /puppypi/alerts/sms_recipients    → JSON list of phone numbers (+65XXXXXXXX)
    """
    _cache      = {}
    _cache_time = 0
    CACHE_TTL   = 300  # 5 minutes

    @classmethod
    def get(cls) -> dict:
        if time.time() - cls._cache_time < cls.CACHE_TTL and cls._cache:
            return cls._cache

        try:
            email_param = ssm.get_parameter(
                Name=f"{SSM_PARAM_PATH}/email_recipients", WithDecryption=False
            )
            sms_param = ssm.get_parameter(
                Name=f"{SSM_PARAM_PATH}/sms_recipients", WithDecryption=False
            )
            cls._cache = {
                "email": json.loads(email_param["Parameter"]["Value"]),
                "sms":   json.loads(sms_param["Parameter"]["Value"]),
            }
        except ClientError as e:
            log.error(f"Failed to load recipients from SSM: {e}")
            # Fall back to hardcoded defaults if SSM unavailable
            cls._cache = {
                "email": ["team@example.com"],
                "sms":   [],
            }

        cls._cache_time = time.time()
        return cls._cache

# ── Rate Limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Simple in-memory rate limiter to prevent alert storms.
    Key is a hash of (alert_type + dedup_key) so same event type
    in the same zone won't spam recipients.
    """
    _last_sent: dict = {}

    @classmethod
    def should_send(cls, alert_type: str, dedup_key: str) -> bool:
        key      = hashlib.md5(f"{alert_type}:{dedup_key}".encode()).hexdigest()
        cooldown = COOLDOWN_SECONDS.get(alert_type, 300)
        last     = cls._last_sent.get(key, 0)

        if time.time() - last < cooldown:
            log.info(f"Rate limited: {alert_type}/{dedup_key} (cooldown {cooldown}s)")
            return False

        cls._last_sent[key] = time.time()
        return True

# ── Email Templates ───────────────────────────────────────────────────────────
def build_html_email(alert_type: str, subject: str, message: str, severity: str) -> str:
    """
    Build a styled HTML email body for the alert.
    Uses inline CSS for maximum email client compatibility.
    """
    color_map = {
        "CRITICAL": "#DC2626",
        "HIGH":     "#EA580C",
        "MEDIUM":   "#D97706",
        "LOW":      "#2563EB",
    }
    badge_color = color_map.get(severity, "#6B7280")
    timestamp   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    icon_map = {
        "INTRUDER": "🚨",
        "HAZARD":   "☣️",
        "SYSTEM":   "⚙️",
    }
    icon = icon_map.get(alert_type, "⚠️")

    # Convert plain message to HTML lines
    message_html = "".join(
        f"<p style='margin:4px 0;color:#374151;font-size:15px;'>{line}</p>"
        for line in message.strip().split("\n") if line.strip()
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <body style="margin:0;padding:0;background:#F3F4F6;font-family:monospace;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding:32px 16px;">
          <table width="580" cellpadding="0" cellspacing="0"
            style="background:#fff;border-radius:12px;overflow:hidden;
                   box-shadow:0 4px 24px rgba(0,0,0,0.08);">

            <!-- Header -->
            <tr><td style="background:#111827;padding:24px 32px;">
              <p style="margin:0;font-size:11px;letter-spacing:4px;color:#6B7280;">
                PUPPYPI SECURITY SYSTEM
              </p>
              <h1 style="margin:8px 0 0;font-size:22px;color:#F9FAFB;font-weight:900;">
                {icon} {subject}
              </h1>
            </td></tr>

            <!-- Severity Badge -->
            <tr><td style="padding:16px 32px 0;">
              <span style="display:inline-block;background:{badge_color};color:#fff;
                           font-size:11px;font-weight:700;letter-spacing:2px;
                           padding:4px 12px;border-radius:4px;">
                {severity} SEVERITY
              </span>
            </td></tr>

            <!-- Message Body -->
            <tr><td style="padding:20px 32px;">
              <div style="background:#F9FAFB;border:1px solid #E5E7EB;
                          border-radius:8px;padding:16px 20px;">
                {message_html}
              </div>
            </td></tr>

            <!-- Footer -->
            <tr><td style="padding:16px 32px 24px;border-top:1px solid #F3F4F6;">
              <p style="margin:0;font-size:11px;color:#9CA3AF;">
                Generated at {timestamp} · PuppyPi Edge Security System
              </p>
              <p style="margin:4px 0 0;font-size:11px;color:#9CA3AF;">
                To manage alert settings, update SSM Parameter Store at
                <code>/puppypi/alerts/</code>
              </p>
            </td></tr>

          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

# ── SES Email Sender ──────────────────────────────────────────────────────────
def send_email(subject: str, message: str, alert_type: str, severity: str):
    """Send HTML alert email to all configured recipients via AWS SES."""
    recipients = RecipientConfig.get().get("email", [])
    if not recipients:
        log.warning("No email recipients configured")
        return

    html_body = build_html_email(alert_type, subject, message, severity)

    try:
        response = ses.send_email(
            Source=SES_SENDER_EMAIL,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Html": {"Data": html_body,  "Charset": "UTF-8"},
                    "Text": {"Data": message,    "Charset": "UTF-8"},
                },
            },
        )
        log.info(f"Email sent to {len(recipients)} recipient(s). MessageId: {response['MessageId']}")
    except ClientError as e:
        log.error(f"SES send failed: {e.response['Error']['Message']}")

# ── SNS SMS Sender ────────────────────────────────────────────────────────────
def send_sms(subject: str, message: str, severity: str):
    """
    Send SMS alert to all configured phone numbers via AWS SNS.
    SMS is kept short — just the key facts, under 160 chars per segment.
    """
    recipients = RecipientConfig.get().get("sms", [])
    if not recipients:
        log.info("No SMS recipients configured, skipping SMS")
        return

    # Build concise SMS (160 char limit per segment)
    # Extract first two lines of the message for brevity
    lines    = [l.strip() for l in message.strip().split("\n") if l.strip()]
    sms_body = f"[PuppyPi {severity}] {subject}\n" + "\n".join(lines[:2])

    if len(sms_body) > 155:
        sms_body = sms_body[:152] + "..."

    for phone in recipients:
        try:
            response = sns.publish(
                PhoneNumber = phone,
                Message     = sms_body,
                MessageAttributes={
                    "AWS.SNS.SMS.SMSType": {
                        "DataType":    "String",
                        "StringValue": "Transactional",  # Ensures high deliverability
                    },
                    "AWS.SNS.SMS.SenderID": {
                        "DataType":    "String",
                        "StringValue": "PuppyPi",        # Appears as sender name
                    },
                },
            )
            log.info(f"SMS sent to {phone}. MessageId: {response['MessageId']}")
        except ClientError as e:
            log.error(f"SNS SMS to {phone} failed: {e.response['Error']['Message']}")

# ── Main Alert Router ─────────────────────────────────────────────────────────
class AlertRouter:
    """
    Central alert dispatcher.
    Called by mqtt_ingestion.py whenever a critical event occurs.

    Usage:
        router = AlertRouter()
        router.send_alert(
            alert_type = "INTRUDER",
            subject    = "Intruder Detected",
            message    = "Zone A, confidence 92%, 14:32 UTC",
            severity   = "HIGH",
            dedup_key  = "zone-A",    # optional: for rate limiting per zone
        )
    """

    def send_alert(
        self,
        alert_type: str,
        subject:    str,
        message:    str,
        severity:   str,
        dedup_key:  Optional[str] = None,
    ):
        """
        Route an alert to the appropriate channels based on severity.

        Args:
            alert_type : "INTRUDER", "HAZARD", or "SYSTEM"
            subject    : Short alert title (used as email subject + SMS header)
            message    : Full alert detail (newline-separated key-value pairs)
            severity   : "CRITICAL", "HIGH", "MEDIUM", or "LOW"
            dedup_key  : Optional key to scope rate limiting (e.g. zone name)
        """
        key = dedup_key or alert_type

        # Rate limit check
        if not RateLimiter.should_send(alert_type, key):
            return

        log.info(f"Routing alert: type={alert_type}, severity={severity}")

        # CRITICAL / HIGH → SMS + Email
        if severity in ("CRITICAL", "HIGH"):
            send_sms(subject, message, severity)
            send_email(subject, message, alert_type, severity)

        # MEDIUM / LOW → Email only
        else:
            send_email(subject, message, alert_type, severity)

        log.info(f"Alert dispatched: {alert_type} [{severity}]")


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick test — sends a test alert to verify SES/SNS setup."""
    logging.basicConfig(level=logging.INFO)
    router = AlertRouter()

    router.send_alert(
        alert_type = "INTRUDER",
        subject    = "TEST — Intruder Detected",
        message    = (
            "Zone: Main Entrance\n"
            "Confidence: 94.2%\n"
            "Time: 2025-01-01T12:00:00Z\n"
            "Snapshot: s3://puppypi-events/snapshots/test.jpg"
        ),
        severity   = "HIGH",
        dedup_key  = "test-zone",
    )
    print("Test alert sent. Check your email and phone.")