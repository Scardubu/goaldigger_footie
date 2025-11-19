import matplotlib.pyplot as plt

def plot_team_performance(data, team):
    if 'league_table' not in data or data['league_table'].empty:
        raise ValueError("League table data is missing or empty.")
    if team not in data['league_table']['team.name'].values:
        raise ValueError(f"Team '{team}' not found in league table data.")

    fig, ax = plt.subplots()
    team_data = data['league_table'][data['league_table']['team.name'] == team]
    ax.bar(['Wins', 'Draws', 'Losses'], [team_data['won'].values[0], team_data['draw'].values[0], team_data['lost'].values[0]])
    ax.set_title(f"{team} Performance")
    return fig