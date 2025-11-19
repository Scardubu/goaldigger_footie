#!/usr/bin/env python
"""
Team Analyzer CLI Tool for GoalDiggers Platform

This script provides a command-line interface for users to analyze specific teams
and get betting insights for their upcoming matches.

Usage:
  python scripts/analyze_team.py --team "Manchester United" --output insights.json
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.setup_env import setup_environment

# Configure environment first
logger = setup_environment()

from scripts.betting_insights import BettingInsightsGenerator


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Get betting insights for specific teams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--team", type=str, help="Team name to analyze")
    parser.add_argument("--list-teams", action="store_true", help="List all available teams")
    parser.add_argument("--output", type=str, help="Path to save results to (JSON)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look ahead")
    parser.add_argument("--model-uri", type=str, help="URI of the model to use for predictions")
    args = parser.parse_args()
    
    # Create insights generator
    insights_generator = BettingInsightsGenerator(model_uri=args.model_uri)
    
    if args.list_teams:
        # List all teams in the database
        with insights_generator.db.session_scope() as session:
            from database.schema import Team
            teams = session.query(Team).order_by(Team.name).all()
            
            if not teams:
                logger.error("No teams found in database")
                return 1
                
            print("Available teams:")
            for i, team in enumerate(teams, 1):
                print(f"{i:3}. {team.name}")
                
            return 0
    
    if not args.team:
        logger.error("Please specify a team name with --team or use --list-teams to see available teams")
        return 1
        
    # Generate insights for the specified team
    team_names = [args.team]
    logger.info(f"Generating insights for team: {args.team}")
    
    results = insights_generator.analyze_user_selected_matches(
        team_names=team_names,
        save_path=args.output
    )
    
    if not results:
        logger.warning(f"No upcoming matches found for {args.team}")
        return 1
        
    logger.info(f"Generated insights for {len(results)} matches")
    
    # Print results to console
    print(f"\nUpcoming matches for {args.team}:")
    print("-" * 50)
    
    for i, insight in enumerate(results, 1):
        home = insight.get('home_team', 'Unknown')
        away = insight.get('away_team', 'Unknown')
        match_date = insight.get('match_date', 'Unknown date')
        
        print(f"{i}. {home} vs {away} on {match_date}")
        
        if insight.get('recommendations'):
            print("  Recommendations:")
            for j, rec in enumerate(insight['recommendations'], 1):
                if 'odds' in rec and 'expected_value' in rec:
                    print(f"   {j}. {rec['bet'].replace('_', ' ').title()} @ {rec['odds']:.2f} (EV: {rec['expected_value']*100:.1f}%)")
                    print(f"      Reason: {rec['reason']}")
                else:
                    print(f"   {j}. {rec['bet'].replace('_', ' ').title()} - {rec.get('reason', 'No reason provided')}")
        else:
            print("  No strong betting recommendations found for this match.")
            
        print()
    
    if args.output:
        print(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
