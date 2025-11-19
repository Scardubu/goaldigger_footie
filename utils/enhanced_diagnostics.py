"""
Enhanced Diagnostic Messages - User-friendly, actionable error and warning messages
Replaces vague warnings with clear explanations and step-by-step solutions
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DiagnosticMessage:
    """Structured diagnostic message with severity and recommendations"""
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    description: str
    impact: str
    recommendations: List[str]
    documentation_link: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Format as Markdown for dashboard display"""
        emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🔴',
        }
        
        lines = [
            f"{emoji.get(self.severity, '❓')} **{self.title}**",
            "",
            self.description,
            "",
            f"**Impact:** {self.impact}",
            "",
            "**How to Fix:**",
        ]
        
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        if self.documentation_link:
            lines.extend(["", f"[📖 Documentation]({self.documentation_link})"])
        
        return "\n".join(lines)
    
    def to_console(self) -> str:
        """Format for console/terminal display"""
        emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🔴',
        }
        
        lines = [
            f"{emoji.get(self.severity, '❓')} {self.title}",
            "",
            f"   {self.description}",
            "",
            f"   Impact: {self.impact}",
            "",
            "   How to Fix:",
        ]
        
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"   {i}. {rec}")
        
        if self.documentation_link:
            lines.extend(["", f"   Documentation: {self.documentation_link}"])
        
        return "\n".join(lines)


class DiagnosticMessageGenerator:
    """Generate user-friendly diagnostic messages for common issues"""
    
    @staticmethod
    def low_real_data_coverage(
        quality_score: float,
        missing_apis: List[str],
        fallback_usage: Dict[str, bool]
    ) -> DiagnosticMessage:
        """Generate message for low real-data coverage"""
        
        # Analyze the situation
        has_critical_missing = 'FOOTBALL_DATA_API_KEY' in missing_apis
        has_fallbacks = any(fallback_usage.values())
        
        if has_critical_missing:
            title = "API Credentials Required for Live Data"
            description = (
                f"Real-time data quality: {quality_score:.0%} (currently using fallback data for some features). "
                f"Your predictions are still accurate using historical data and cached information, "
                f"but live standings and current fixtures require API access."
            )
            impact = (
                "Predictions use historical patterns (excellent) but lack real-time standings "
                "and live fixture updates. Quality score could improve to 90%+ with API credentials."
            )
            
            recommendations = [
                "**Get Football-Data.org API Key** (Free tier available):\n"
                "   • Visit: https://www.football-data.org/client/register\n"
                "   • Register for free account (10 requests/minute)\n"
                "   • Copy your API key from dashboard",
                
                "**Configure Environment Variable:**\n"
                "   • Create/edit `.env` file in project root\n"
                "   • Add line: `FOOTBALL_DATA_API_KEY=your_key_here`\n"
                "   • Save file",
                
                "**Restart Application:**\n"
                "   • Stop current instance (Ctrl+C)\n"
                "   • Run: `python unified_launcher.py dashboard`\n"
                "   • Verify \"✅ API credentials configured\" in startup logs",
                
                "**Verify Improvement:**\n"
                "   • Check logs for \"[REAL DATA] Quality score: 0.9x\"\n"
                "   • Live standings will show current data\n"
                "   • Fixture times will be real-time"
            ]
            
            return DiagnosticMessage(
                severity='warning',
                title=title,
                description=description,
                impact=impact,
                recommendations=recommendations,
                documentation_link="https://github.com/yourusername/footie/blob/main/docs/API_SETUP.md"
            )
        
        elif has_fallbacks:
            title = "Using Fallback Data for Some Features"
            description = (
                f"Real-time data quality: {quality_score:.0%}. "
                f"Some features are using fallback/cached data. "
                f"Predictions remain accurate but may not reflect very recent changes."
            )
            impact = "Minor - Recent form and standings may be slightly outdated (typically < 24 hours)"
            
            recommendations = [
                "**Optional: Add Additional API Sources for Redundancy:**\n"
                "   • API-Football: https://www.api-football.com/\n"
                "   • Set API_FOOTBALL_KEY in .env\n"
                "   • Provides fallback if primary API is unavailable",
                
                "**Check API Rate Limits:**\n"
                "   • Football-Data.org Free tier: 10 requests/minute\n"
                "   • Consider upgrading if hitting limits frequently\n"
                "   • Monitor logs for rate limit messages"
            ]
            
            return DiagnosticMessage(
                severity='info',
                title=title,
                description=description,
                impact=impact,
                recommendations=recommendations,
            )
        
        else:
            # No issues
            title = "Real-Time Data Integration Healthy"
            description = (
                f"Real-time data quality: {quality_score:.0%}. "
                f"All API integrations operational. Predictions use both historical "
                f"database records and current live data."
            )
            impact = "None - System operating optimally"
            
            recommendations = [
                "No action needed. System is healthy.",
                "Consider adding optional UNDERSTAT_API_KEY for enhanced xG stats."
            ]
            
            return DiagnosticMessage(
                severity='info',
                title=title,
                description=description,
                impact=impact,
                recommendations=recommendations,
            )
    
    @staticmethod
    def missing_api_key(api_name: str, is_critical: bool = True) -> DiagnosticMessage:
        """Generate message for missing API key"""
        
        api_info = {
            'FOOTBALL_DATA_API_KEY': {
                'name': 'Football-Data.org',
                'url': 'https://www.football-data.org/client/register',
                'features': 'Live fixtures, current standings, team statistics',
                'tier': 'Free (10 req/min)',
            },
            'API_FOOTBALL_KEY': {
                'name': 'API-Football',
                'url': 'https://www.api-football.com/',
                'features': 'Backup data source, additional stats',
                'tier': 'Free (100 req/day)',
            },
            'UNDERSTAT_API_KEY': {
                'name': 'Understat',
                'url': 'https://understat.com/',
                'features': 'Enhanced xG statistics, shot data',
                'tier': 'Optional premium',
            },
        }
        
        info = api_info.get(api_name, {
            'name': api_name,
            'url': 'Unknown',
            'features': 'Various features',
            'tier': 'Unknown',
        })
        
        severity = 'warning' if is_critical else 'info'
        title = f"Missing API Key: {info['name']}"
        
        description = (
            f"The {info['name']} API key is not configured. "
            f"This affects: {info['features']}."
        )
        
        if is_critical:
            impact = f"Critical features unavailable. System using fallback data."
        else:
            impact = f"Optional features unavailable. Core functionality unaffected."
        
        recommendations = [
            f"**Register for API Access:**\n"
            f"   • Visit: {info['url']}\n"
            f"   • Tier: {info['tier']}\n"
            f"   • Sign up and get API key",
            
            f"**Add to Environment:**\n"
            f"   • Open .env file\n"
            f"   • Add: {api_name}=your_key_here\n"
            f"   • Save and restart application"
        ]
        
        return DiagnosticMessage(
            severity=severity,
            title=title,
            description=description,
            impact=impact,
            recommendations=recommendations,
        )
    
    @staticmethod
    def database_connection_error(db_type: str, fallback_used: bool = False) -> DiagnosticMessage:
        """Generate message for database connection issues"""
        
        if fallback_used:
            title = f"Using SQLite Fallback (PostgreSQL Unavailable)"
            description = (
                f"Could not connect to PostgreSQL database. "
                f"Application has automatically switched to SQLite fallback. "
                f"All core features remain operational."
            )
            severity = 'warning'
            impact = "Minor - Using local SQLite instead of PostgreSQL. Performance may be slightly reduced for large queries."
            
            recommendations = [
                "**Verify PostgreSQL is Running:**\n"
                "   • Check if PostgreSQL service is started\n"
                "   • Windows: Services > PostgreSQL\n"
                "   • Command: `pg_isready -h localhost -p 5432`",
                
                "**Check Connection Settings:**\n"
                "   • Verify DATABASE_URL in .env\n"
                "   • Format: postgresql://user:pass@localhost:5432/dbname\n"
                "   • Ensure credentials are correct",
                
                "**Optional: Continue with SQLite:**\n"
                "   • SQLite fallback is fully functional\n"
                "   • Suitable for development and moderate usage\n"
                "   • No action needed if performance acceptable"
            ]
        else:
            title = f"Database Connection Failed"
            description = f"Could not connect to {db_type} database and no fallback available."
            severity = 'error'
            impact = "Critical - Application cannot access data storage"
            
            recommendations = [
                "**Check Database Service:**\n"
                "   • Ensure database is running\n"
                "   • Verify connection settings in .env",
                
                "**Contact Administrator:**\n"
                "   • Database may need configuration\n"
                "   • Check logs for detailed error messages"
            ]
        
        return DiagnosticMessage(
            severity=severity,
            title=title,
            description=description,
            impact=impact,
            recommendations=recommendations,
        )
    
    @staticmethod
    def rate_limit_warning(api_name: str, retry_after: int) -> DiagnosticMessage:
        """Generate message for API rate limit issues"""
        
        title = f"API Rate Limit Reached: {api_name}"
        description = (
            f"The {api_name} API rate limit has been reached. "
            f"The system is automatically retrying with exponential backoff. "
            f"Retry after: {retry_after} seconds."
        )
        severity = 'info'
        impact = "Temporary - Data fetching delayed but will complete"
        
        recommendations = [
            "**No Immediate Action Required:**\n"
            "   • System handling retries automatically\n"
            "   • Data will be fetched when limit resets",
            
            "**Consider for Future:**\n"
            "   • Upgrade API tier for higher rate limits\n"
            "   • Implement request caching (already enabled)\n"
            "   • Reduce concurrent request parallelism",
            
            "**Current Limits:**\n"
            "   • Football-Data.org Free: 10 req/min\n"
            "   • API-Football Free: 100 req/day\n"
            "   • Check API provider for upgrade options"
        ]
        
        return DiagnosticMessage(
            severity=severity,
            title=title,
            description=description,
            impact=impact,
            recommendations=recommendations,
        )


# Convenience functions for common scenarios
def get_low_coverage_message() -> str:
    """Get improved low coverage message (replaces old vague warning)"""
    missing_apis = []
    if not os.getenv('FOOTBALL_DATA_API_KEY'):
        missing_apis.append('FOOTBALL_DATA_API_KEY')
    
    generator = DiagnosticMessageGenerator()
    message = generator.low_real_data_coverage(
        quality_score=0.75,  # Typical score without API
        missing_apis=missing_apis,
        fallback_usage={'standings': True, 'fixtures': False}
    )
    
    return message.to_console()


def get_compact_coverage_warning() -> str:
    """Get compact single-line warning for dashboard"""
    if not os.getenv('FOOTBALL_DATA_API_KEY'):
        return (
            "⚠️ API credentials missing — Set FOOTBALL_DATA_API_KEY for live data. "
            "Run: python utils/api_health_checker.py for setup guide"
        )
    
    return "✅ Real-time data integration healthy"


if __name__ == '__main__':
    # Demo the improved messages
    print("\n" + "="*70)
    print("Enhanced Diagnostic Messages Demo")
    print("="*70 + "\n")
    
    # Example 1: Low coverage
    print("Example 1: Low Real-Data Coverage")
    print("-" * 70)
    print(get_low_coverage_message())
    print("\n")
    
    # Example 2: Missing API key
    print("Example 2: Missing API Key")
    print("-" * 70)
    generator = DiagnosticMessageGenerator()
    msg = generator.missing_api_key('FOOTBALL_DATA_API_KEY', is_critical=True)
    print(msg.to_console())
    print("\n")
    
    # Example 3: Database fallback
    print("Example 3: Database Fallback")
    print("-" * 70)
    msg = generator.database_connection_error('PostgreSQL', fallback_used=True)
    print(msg.to_console())
    print("\n")
