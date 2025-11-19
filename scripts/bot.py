import logging
import time
from typing import Dict

import schedule
import telegram

from scripts.model import HybridPredictor

logger = logging.getLogger(__name__)


class BettingBot:
    def __init__(self):
        self.bot = telegram.Bot(token=os.getenv("TELEGRAM_TOKEN"))
        self.predictor = HybridPredictor()
        self.last_alert = {}

    def _value_bet_condition(self, prediction: Dict) -> bool:
        """Identify bets with positive expected value"""
        return (
            prediction["confidence"] > 0.7
            and prediction["home_win"] > 0.5
            and (1 / prediction["home_win"]) > 2.0
        )

    def _arbitrage_check(self, odds: List[float]) -> bool:
        """Detect arbitrage opportunities"""
        total = sum(1 / odd for odd in odds)
        return total < 0.95

    def _generate_insight(self, match: Dict) -> str:
        """AI-powered match analysis"""
        prediction = self.predictor.predict(match)

        insights = []
        if self._value_bet_condition(prediction):
            insights.append("🔥 HIGH-VALUE BET")
        if self._arbitrage_check(match["odds"]):
            insights.append("💎 ARBITRAGE OPPORTUNITY")

        return "\\n".join(insights) if insights else None

    def send_alerts(self):
        """Automated decision-making pipeline"""
        try:
            with open("data/processed/valid_matches.json") as f:
                matches = json.load(f)

            for match in matches:
                insight = self._generate_insight(match)
                if insight and match != self.last_alert.get("id"):
                    message = f"""
                    🚨 {match['teams'][0]} vs {match['teams'][1]}
                    📊 {insight}
                    🎯 Confidence: {prediction['confidence']*100:.1f}%
                    ⏱ Last Updated: {datetime.now().strftime('%H:%M')}
                    """
                    self.bot.send_message(chat_id=os.getenv("CHAT_ID"), text=message)
                    self.last_alert = match
        except Exception as e:
            logger.exception("Error in send_alerts: %s", e)


if __name__ == "__main__":
    bot = BettingBot()
    schedule.every(10).minutes.do(bot.send_alerts)

    while True:
        schedule.run_pending()
        time.sleep(60)
