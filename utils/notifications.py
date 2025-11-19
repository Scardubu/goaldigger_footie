import logging
import os

import requests

# Optional Slack dependency guard
try:
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore
    _SLACK_AVAILABLE = True
except Exception:  # broad to catch partial install issues
    WebClient = None  # type: ignore
    SlackApiError = Exception  # fallback sentinel
    _SLACK_AVAILABLE = False

from dashboard.error_log import log_error  # Import log_error

logger = logging.getLogger(__name__)

def send_slack_message(text: str, file_path: str = None):
    """Send a Slack message if slack_sdk is available; otherwise log a no-op.

    This graceful degradation prevents test failures when slack_sdk or its
    transitive deps are absent or partially installed.
    """
    if not _SLACK_AVAILABLE:
        logger.info("Slack SDK not available - skipping Slack notification (no-op).")
        return
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL")
    if not slack_token or not channel:
        logger.debug("Slack credentials missing; skipping send.")
        return
    client = WebClient(token=slack_token)  # type: ignore
    try:
        client.chat_postMessage(channel=channel, text=text)
        logger.info(f"Slack message sent to channel {channel}.")
        if file_path:
            try:
                with open(file_path, "rb") as f:
                    client.files_upload_v2(
                        channel=channel,
                        file=f,
                        filename=os.path.basename(file_path),
                        initial_comment=f"Attached file: {os.path.basename(file_path)}"
                    )
                logger.info(f"Slack file '{os.path.basename(file_path)}' uploaded to {channel}.")
            except FileNotFoundError:
                log_error(f"Slack file upload failed: File not found at {file_path}", None)
            except Exception as file_e:
                log_error(f"Slack file upload failed for {file_path}", file_e)
    except SlackApiError as e:  # type: ignore
        # Some versions may not carry response attr; guard access
        err_msg = getattr(getattr(e, 'response', {}), 'data', {}).get('error', str(e))
        log_error(f"Slack API error sending message/file: {err_msg}", e)
    except Exception as e:
        log_error("Unexpected error sending Slack notification", e)


def send_telegram_message(text: str, file_path: str = None):
    """Sends a message and optionally a document to a configured Telegram chat."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) not set. Skipping Telegram notification.")
        return

    # Send text message
    url_message = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data_message = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url_message, data=data_message, timeout=10) # Add timeout
        response.raise_for_status() # Check for HTTP errors
        logger.info(f"Telegram message sent successfully to chat ID {chat_id}.")
    except requests.exceptions.RequestException as e:
        log_error("Telegram API request failed sending message", e) # Use log_error
    except Exception as e:
        log_error("Unexpected error sending Telegram message", e) # Use log_error


    # Send file if path provided
    if file_path:
        url_document = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        try:
            with open(file_path, "rb") as f:
                files = {"document": (os.path.basename(file_path), f)}
                data_document = {"chat_id": chat_id}
                response = requests.post(url_document, data=data_document, files=files, timeout=30) # Longer timeout for files
                response.raise_for_status() # Check for HTTP errors
            logger.info(f"Telegram document '{os.path.basename(file_path)}' sent successfully to chat ID {chat_id}.")
        except FileNotFoundError:
            log_error(f"Telegram file upload failed: File not found at {file_path}", None) # Use log_error, pass None for exception if not available
        except requests.exceptions.RequestException as e:
            log_error(f"Telegram API request failed sending document {file_path}", e) # Use log_error
        except Exception as e:
            log_error(f"Unexpected error sending Telegram document {file_path}", e) # Use log_error
