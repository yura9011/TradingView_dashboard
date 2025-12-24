"""
Telegram Notifier - Send trading signals to Telegram.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trading signals and reports to Telegram."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (or TELEGRAM_BOT_TOKEN env var)
            chat_id: Chat ID to send messages (or TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN required")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID required")
        
        self.bot = Bot(token=self.bot_token)
        logger.info("Telegram notifier initialized")
    
    async def send_message(self, text: str, parse_mode: str = ParseMode.MARKDOWN) -> bool:
        """Send a text message.
        
        Args:
            text: Message text
            parse_mode: Telegram parse mode
            
        Returns:
            True if sent successfully
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            logger.info("Message sent to Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def send_photo(
        self,
        photo_path: str,
        caption: Optional[str] = None,
    ) -> bool:
        """Send a photo with optional caption.
        
        Args:
            photo_path: Path to image file
            caption: Optional caption
            
        Returns:
            True if sent successfully
        """
        photo_path = Path(photo_path)
        if not photo_path.exists():
            logger.error(f"Photo not found: {photo_path}")
            return False
        
        try:
            with open(photo_path, "rb") as photo:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=caption[:1024] if caption else None,  # Telegram limit
                    parse_mode=ParseMode.MARKDOWN,
                )
            logger.info("Photo sent to Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            return False
    
    async def send_signal_alert(
        self,
        symbol: str,
        signal_type: str,
        pattern: str,
        confidence: float,
        trend: str,
        summary: str,
        chart_path: Optional[str] = None,
    ) -> bool:
        """Send a complete signal alert.
        
        Args:
            symbol: Ticker symbol
            signal_type: Signal classification
            pattern: Detected pattern
            confidence: Pattern confidence
            trend: Current trend
            summary: Analysis summary
            chart_path: Path to annotated chart
            
        Returns:
            True if sent successfully
        """
        # Build message
        emoji = "✅" if signal_type == "candidate" else ("❌" if signal_type == "not_candidate" else "⏳")
        trend_emoji = "📈" if trend == "up" else ("📉" if trend == "down" else "➡️")
        
        message = f"""
{emoji} *Trading Signal: {symbol}*

*Type:* {signal_type.upper()}
*Pattern:* {pattern.replace('_', ' ').title()}
*Confidence:* {confidence:.0%}
*Trend:* {trend_emoji} {trend.upper()}

_{summary[:200]}_
"""
        
        # Send photo with caption if available
        if chart_path and Path(chart_path).exists():
            return await self.send_photo(chart_path, caption=message)
        else:
            return await self.send_message(message)
    
    def send_signal_sync(
        self,
        symbol: str,
        signal_type: str,
        pattern: str,
        confidence: float,
        trend: str,
        summary: str,
        chart_path: Optional[str] = None,
    ) -> bool:
        """Synchronous wrapper for send_signal_alert."""
        return asyncio.run(
            self.send_signal_alert(
                symbol, signal_type, pattern, confidence, trend, summary, chart_path
            )
        )


def get_telegram_notifier(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> TelegramNotifier:
    """Get TelegramNotifier instance."""
    return TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
