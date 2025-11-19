"""
Improved Diagnostic Message Generator
Provides user-friendly, actionable messages for data quality issues
"""

import os
from typing import Dict, List


def generate_diagnostic_message(quality_score: float = None, real_data_used: bool = None) -> str:
    """
    Generate user-friendly diagnostic message based on data quality
    
    Args:
        quality_score: Real data quality score (0.0-1.0)
        real_data_used: Whether real data was used in predictions
    
    Returns:
        Formatted diagnostic message with recommendations
    """
    # Check API credentials
    football_data_key = os.getenv('FOOTBALL_DATA_API_KEY')
    api_football_key = os.getenv('API_FOOTBALL_KEY')
    understat_key = os.getenv('UNDERSTAT_API_KEY')
    
    # Count configured APIs
    apis_configured = sum([
        bool(football_data_key),
        bool(api_football_key),
        bool(understat_key)
    ])
    
    messages = []
    emoji = "✅"
    status = "EXCELLENT"
    
    # Determine status based on quality and API configuration
    if quality_score is not None:
        if quality_score >= 0.90:
            emoji = "✅"
            status = "EXCELLENT"
        elif quality_score >= 0.80:
            emoji = "🟢"
            status = "GOOD"
        elif quality_score >= 0.70:
            emoji = "🟡"
            status = "ACCEPTABLE"
        else:
            emoji = "🔴"
            status = "LOW"
    
    # Build diagnostic message
    if not football_data_key and not api_football_key:
        emoji = "🔴"
        status = "CRITICAL"
        messages.append("**No API credentials configured** - Using fallback data only")
        messages.append("")
        messages.append("**Quick Fix:**")
        messages.append("1. Open `.env` file in project root")
        messages.append("2. Add: `FOOTBALL_DATA_API_KEY=your_key_here`")
        messages.append("3. Get free API key: https://www.football-data.org/client/register")
        messages.append("4. Restart the dashboard")
        messages.append("")
        messages.append("**Impact:** Predictions using historical data only (no current form, standings, or recent results)")
        
    elif not football_data_key:
        emoji = "🟡"
        status = "DEGRADED"
        messages.append("**Primary API not configured** - Using fallback API (API-Football)")
        messages.append("")
        messages.append("**Recommendation:** Add Football-Data.org API key for best results:")
        messages.append("• Get free key: https://www.football-data.org/client/register")
        messages.append("• Add to `.env`: `FOOTBALL_DATA_API_KEY=your_key_here`")
        messages.append("• Provides more comprehensive standings and fixture data")
        
    elif apis_configured == 1:
        emoji = "🟢"
        status = "GOOD"
        messages.append("**Single API configured** - Core functionality operational")
        messages.append("")
        messages.append("**Optional Enhancement:** Add API-Football key for redundancy:")
        messages.append("• Provides fallback if primary API has issues")
        messages.append("• Get key: https://www.api-football.com/")
        messages.append("• Add to `.env`: `API_FOOTBALL_KEY=your_key_here`")
        
    elif apis_configured == 2:
        emoji = "✅"
        status = "EXCELLENT"
        messages.append("**Dual APIs configured** - Excellent redundancy!")
        messages.append("")
        messages.append("**Optional:** Add Understat API for enhanced xG statistics")
        messages.append("• Not required but provides additional analytics")
        messages.append("• Currently using calculated xG estimates")
        
    else:
        emoji = "🏆"
        status = "OPTIMAL"
        messages.append("**All APIs configured** - Maximum data coverage! 🎉")
        messages.append("")
        messages.append("You're getting the best possible predictions with:")
        messages.append("• Live standings from Football-Data.org")
        messages.append("• Fallback coverage from API-Football")
        messages.append("• Enhanced xG statistics from Understat")
    
    # Add quality score details if available
    if quality_score is not None:
        messages.append("")
        messages.append(f"**Current Quality Score:** {quality_score:.2f}/1.00 ({_quality_rating(quality_score)})")
        
        if real_data_used:
            messages.append("✅ Real-time data: **ACTIVE**")
        else:
            messages.append("⚠️ Real-time data: **FALLBACK MODE**")
    
    # Header
    header = f"{emoji} **Data Quality Status: {status}**"
    
    return header + "\n\n" + "\n".join(messages)


def generate_compact_diagnostic(quality_score: float = None) -> str:
    """Generate compact one-line diagnostic message"""
    football_data_key = os.getenv('FOOTBALL_DATA_API_KEY')
    api_football_key = os.getenv('API_FOOTBALL_KEY')
    
    if not football_data_key and not api_football_key:
        return "🔴 **No API credentials** - [Configure API keys](#) for live data"
    elif not football_data_key:
        return "🟡 **Fallback API active** - Add FOOTBALL_DATA_API_KEY for optimal performance"
    elif not api_football_key:
        return "🟢 **Primary API configured** - Add API_FOOTBALL_KEY for redundancy"
    else:
        if quality_score and quality_score >= 0.90:
            return "✅ **Optimal configuration** - All systems operational"
        else:
            return "🟢 **APIs configured** - Data quality: Good"


def generate_setup_instructions() -> str:
    """Generate detailed setup instructions for missing APIs"""
    instructions = []
    
    football_data_key = os.getenv('FOOTBALL_DATA_API_KEY')
    api_football_key = os.getenv('API_FOOTBALL_KEY')
    
    if not football_data_key:
        instructions.append("### 📡 Football-Data.org Setup (PRIMARY)")
        instructions.append("")
        instructions.append("1. **Visit:** https://www.football-data.org/client/register")
        instructions.append("2. **Register** for a free account")
        instructions.append("3. **Copy** your API key from the dashboard")
        instructions.append("4. **Open** `.env` file in project root")
        instructions.append("5. **Add** this line:")
        instructions.append("   ```")
        instructions.append("   FOOTBALL_DATA_API_KEY=your_key_here")
        instructions.append("   ```")
        instructions.append("6. **Save** and restart the dashboard")
        instructions.append("")
        instructions.append("**Free Tier Limits:**")
        instructions.append("• 10 requests per minute")
        instructions.append("• Access to 5 major leagues (PL, La Liga, Serie A, Bundesliga, Ligue 1)")
        instructions.append("• Live standings, fixtures, and team statistics")
        instructions.append("")
    
    if not api_football_key:
        instructions.append("### 🔄 API-Football Setup (FALLBACK - Optional)")
        instructions.append("")
        instructions.append("1. **Visit:** https://rapidapi.com/api-sports/api/api-football")
        instructions.append("2. **Subscribe** to free tier")
        instructions.append("3. **Copy** your API key")
        instructions.append("4. **Add** to `.env`:")
        instructions.append("   ```")
        instructions.append("   API_FOOTBALL_KEY=your_key_here")
        instructions.append("   API_FOOTBALL_HOST=api-football-v1.p.rapidapi.com")
        instructions.append("   ```")
        instructions.append("5. **Save** and restart")
        instructions.append("")
        instructions.append("**Free Tier Limits:**")
        instructions.append("• 100 requests per day")
        instructions.append("• Access to all major leagues")
        instructions.append("• Provides redundancy if Football-Data.org is down")
        instructions.append("")
    
    if not instructions:
        return "✅ **All recommended APIs configured!** You're all set."
    
    return "\n".join(instructions)


def _quality_rating(score: float) -> str:
    """Convert quality score to rating"""
    if score >= 0.90:
        return "Excellent"
    elif score >= 0.80:
        return "Good"
    elif score >= 0.70:
        return "Acceptable"
    elif score >= 0.60:
        return "Fair"
    else:
        return "Poor"


def get_api_status_summary() -> Dict:
    """Get summary of API configuration status"""
    return {
        'football_data': {
            'configured': bool(os.getenv('FOOTBALL_DATA_API_KEY')),
            'name': 'Football-Data.org',
            'type': 'primary',
            'importance': 'critical',
        },
        'api_football': {
            'configured': bool(os.getenv('API_FOOTBALL_KEY')),
            'name': 'API-Football',
            'type': 'fallback',
            'importance': 'recommended',
        },
        'understat': {
            'configured': bool(os.getenv('UNDERSTAT_API_KEY')),
            'name': 'Understat',
            'type': 'enhancement',
            'importance': 'optional',
        },
    }


# CLI interface for testing
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*70)
    print("🏥 GoalDiggers API Diagnostic Report")
    print("="*70 + "\n")
    
    # Show API status
    status = get_api_status_summary()
    print("📡 **API Configuration:**\n")
    for api_key, api_info in status.items():
        emoji = "✅" if api_info['configured'] else "❌"
        print(f"   {emoji} {api_info['name']} ({api_info['importance']})")
    
    print("\n" + "-"*70 + "\n")
    
    # Show diagnostic message (simulating quality score of 0.85)
    print(generate_diagnostic_message(quality_score=0.85, real_data_used=True))
    
    print("\n" + "-"*70 + "\n")
    
    # Show setup instructions if needed
    instructions = generate_setup_instructions()
    if "all set" not in instructions.lower():
        print(instructions)
    
    print("="*70 + "\n")
