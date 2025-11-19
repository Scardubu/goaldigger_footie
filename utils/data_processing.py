import pandas as pd

def process_league_data(data):
    # Process the raw API data into useful formats
    league_table = pd.DataFrame(data['standings'][0]['table'])
    top_scorers = pd.DataFrame(data['scorers'])
    teams = league_table['team.name'].tolist()

    return {
        'league_table': pd.DataFrame(data['standings'][0]['table']),
        'top_scorers': pd.DataFrame(data['scorers']),
        'teams': teams
    }

def process_team_data(data):
    # Process team-specific data
    team_info = {
        'name': data['name'],
        'venue': data['venue'],
        'founded': data['founded'],
        'squad': pd.DataFrame(data['squad']),
    }
    return team_info