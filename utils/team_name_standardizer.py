#!/usr/bin/env python3
"""
Team Name Standardization System

Ensures consistent team naming across the entire platform for easy search and display.
Provides canonical names, aliases, and search optimization.

Phase 6 Integration: Standardize all team references.
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Canonical team names with comprehensive aliases
CANONICAL_TEAM_NAMES = {
    # Premier League
    "Manchester City": {
        "aliases": ["Man City", "MCFC", "Man. City", "Manchester C.", "City"],
        "search_terms": ["manchester", "city", "mcfc", "guardiola", "etihad"],
        "short": "MCI",
        "country": "England",
        "league": "Premier League"
    },
    "Arsenal": {
        "aliases": ["Arsenal FC", "AFC", "The Gunners", "Ars"],
        "search_terms": ["arsenal", "gunners", "arteta", "emirates"],
        "short": "ARS",
        "country": "England",
        "league": "Premier League"
    },
    "Liverpool": {
        "aliases": ["Liverpool FC", "LFC", "The Reds", "Liv"],
        "search_terms": ["liverpool", "reds", "klopp", "anfield"],
        "short": "LIV",
        "country": "England",
        "league": "Premier League"
    },
    "Chelsea": {
        "aliases": ["Chelsea FC", "CFC", "The Blues", "Che"],
        "search_terms": ["chelsea", "blues", "stamford bridge"],
        "short": "CHE",
        "country": "England",
        "league": "Premier League"
    },
    "Tottenham Hotspur": {
        "aliases": ["Tottenham", "Spurs", "THFC", "Tot"],
        "search_terms": ["tottenham", "spurs", "hotspur", "white hart lane"],
        "short": "TOT",
        "country": "England",
        "league": "Premier League"
    },
    "Manchester United": {
        "aliases": ["Man United", "MUFC", "Man. United", "United", "ManUtd"],
        "search_terms": ["manchester", "united", "mufc", "old trafford", "red devils"],
        "short": "MUN",
        "country": "England",
        "league": "Premier League"
    },
    "Newcastle United": {
        "aliases": ["Newcastle", "NUFC", "The Magpies", "New"],
        "search_terms": ["newcastle", "magpies", "toon", "st james"],
        "short": "NEW",
        "country": "England",
        "league": "Premier League"
    },
    "Aston Villa": {
        "aliases": ["Villa", "AVFC", "The Villans", "AVL"],
        "search_terms": ["aston", "villa", "villans", "villa park"],
        "short": "AVL",
        "country": "England",
        "league": "Premier League"
    },
    "Brighton & Hove Albion": {
        "aliases": ["Brighton", "BHAFC", "Seagulls", "BHA"],
        "search_terms": ["brighton", "hove", "seagulls", "amex"],
        "short": "BHA",
        "country": "England",
        "league": "Premier League"
    },
    "West Ham United": {
        "aliases": ["West Ham", "WHU", "The Hammers", "WHM"],
        "search_terms": ["west ham", "hammers", "london stadium"],
        "short": "WHU",
        "country": "England",
        "league": "Premier League"
    },
    "Nottingham Forest": {
        "aliases": ["Nottingham Forest FC", "NFFC", "Forest", "Notts Forest"],
        "search_terms": ["nottingham", "forest", "city ground"],
        "short": "NFO",
        "country": "England",
        "league": "Premier League"
    },
    "Crystal Palace": {
        "aliases": ["Crystal Palace FC", "CPFC", "Palace", "The Eagles"],
        "search_terms": ["crystal", "palace", "selhurst park"],
        "short": "CRY",
        "country": "England",
        "league": "Premier League"
    },
    
    # La Liga
    "Real Madrid": {
        "aliases": ["Madrid", "Real", "RM", "RMA", "Los Blancos"],
        "search_terms": ["real", "madrid", "blancos", "bernabeu", "galacticos"],
        "short": "RMA",
        "country": "Spain",
        "league": "La Liga"
    },
    "Barcelona": {
        "aliases": ["Barça", "Barca", "FCB", "FC Barcelona", "Blaugrana"],
        "search_terms": ["barcelona", "barca", "barça", "camp nou", "blaugrana"],
        "short": "BAR",
        "country": "Spain",
        "league": "La Liga"
    },
    "Atletico Madrid": {
        "aliases": ["Atletico", "Atleti", "ATM", "Los Rojiblancos"],
        "search_terms": ["atletico", "atleti", "rojiblancos", "wanda"],
        "short": "ATM",
        "country": "Spain",
        "league": "La Liga"
    },
    "Sevilla": {
        "aliases": ["Sevilla FC", "SFC", "SEV"],
        "search_terms": ["sevilla", "nervion"],
        "short": "SEV",
        "country": "Spain",
        "league": "La Liga"
    },
    "Valencia": {
        "aliases": ["Valencia CF", "VCF", "VAL", "Los Che"],
        "search_terms": ["valencia", "mestalla"],
        "short": "VAL",
        "country": "Spain",
        "league": "La Liga"
    },
    "Real Sociedad": {
        "aliases": ["Sociedad", "Real Soc.", "RSO", "La Real"],
        "search_terms": ["real sociedad", "sociedad", "anoeta"],
        "short": "RSO",
        "country": "Spain",
        "league": "La Liga"
    },
    
    # Bundesliga
    "Bayern Munich": {
        "aliases": ["Bayern", "FC Bayern", "Die Roten"],
        "search_terms": ["bayern", "munich", "munchen", "allianz arena"],
        "short": "BAY",
        "country": "Germany",
        "league": "Bundesliga"
    },
    "Borussia Dortmund": {
        "aliases": ["Dortmund", "BVB", "Die Schwarzgelben"],
        "search_terms": ["borussia", "dortmund", "bvb", "signal iduna"],
        "short": "BVB",
        "country": "Germany",
        "league": "Bundesliga"
    },
    "RB Leipzig": {
        "aliases": ["Leipzig", "RBL", "Die Roten Bullen"],
        "search_terms": ["leipzig", "red bull"],
        "short": "RBL",
        "country": "Germany",
        "league": "Bundesliga"
    },
    "Bayer Leverkusen": {
        "aliases": ["Leverkusen", "B04", "Die Werkself"],
        "search_terms": ["bayer", "leverkusen", "bayarena"],
        "short": "B04",
        "country": "Germany",
        "league": "Bundesliga"
    },
    
    # Serie A
    "Inter Milan": {
        "aliases": ["Inter", "Internazionale", "FCIM", "Nerazzurri"],
        "search_terms": ["inter", "milan", "internazionale", "nerazzurri", "san siro"],
        "short": "INT",
        "country": "Italy",
        "league": "Serie A"
    },
    "AC Milan": {
        "aliases": ["Milan", "ACM", "Rossoneri"],
        "search_terms": ["ac milan", "milan", "rossoneri", "san siro"],
        "short": "MIL",
        "country": "Italy",
        "league": "Serie A"
    },
    "Juventus": {
        "aliases": ["Juve", "JUV", "La Vecchia Signora", "Bianconeri"],
        "search_terms": ["juventus", "juve", "bianconeri", "allianz stadium"],
        "short": "JUV",
        "country": "Italy",
        "league": "Serie A"
    },
    "Napoli": {
        "aliases": ["SSC Napoli", "NAP", "Gli Azzurri"],
        "search_terms": ["napoli", "naples", "maradona"],
        "short": "NAP",
        "country": "Italy",
        "league": "Serie A"
    },
    "AS Roma": {
        "aliases": ["Roma", "ASR", "I Giallorossi"],
        "search_terms": ["roma", "rome", "giallorossi", "olimpico"],
        "short": "ROM",
        "country": "Italy",
        "league": "Serie A"
    },
    
    # Ligue 1
    "Paris Saint-Germain": {
        "aliases": ["PSG", "Paris SG", "Les Parisiens"],
        "search_terms": ["psg", "paris", "saint-germain", "parc des princes"],
        "short": "PSG",
        "country": "France",
        "league": "Ligue 1"
    },
    "Marseille": {
        "aliases": ["OM", "Olympique Marseille", "Les Phocéens"],
        "search_terms": ["marseille", "velodrome"],
        "short": "MAR",
        "country": "France",
        "league": "Ligue 1"
    },
    "Lyon": {
        "aliases": ["OL", "Olympique Lyon", "Les Gones"],
        "search_terms": ["lyon", "groupama stadium"],
        "short": "LYO",
        "country": "France",
        "league": "Ligue 1"
    },
    "Monaco": {
        "aliases": ["AS Monaco", "ASM", "Les Monégasques"],
        "search_terms": ["monaco", "louis ii"],
        "short": "MON",
        "country": "France",
        "league": "Ligue 1"
    }
}


class TeamNameStandardizer:
    """Standardize team names across the platform."""
    
    def __init__(self):
        self.canonical_names = CANONICAL_TEAM_NAMES
        self._alias_map = self._build_alias_map()
        self._search_index = self._build_search_index()
        
    def _build_alias_map(self) -> Dict[str, str]:
        """Build mapping from all aliases to canonical names."""
        alias_map = {}
        
        for canonical_name, data in self.canonical_names.items():
            # Map canonical name to itself
            alias_map[canonical_name.lower()] = canonical_name
            
            # Map all aliases to canonical name
            for alias in data["aliases"]:
                alias_map[alias.lower()] = canonical_name
                
        return alias_map
    
    def _build_search_index(self) -> Dict[str, Set[str]]:
        """Build search index for fuzzy team name matching."""
        search_index = {}
        
        for canonical_name, data in self.canonical_names.items():
            for term in data["search_terms"]:
                term_lower = term.lower()
                if term_lower not in search_index:
                    search_index[term_lower] = set()
                search_index[term_lower].add(canonical_name)
                
        return search_index
    
    def standardize(self, team_name: str) -> str:
        """
        Convert any team name variant to canonical form.
        
        Args:
            team_name: Raw team name (can be alias, short form, etc.)
            
        Returns:
            Canonical team name
        """
        if not team_name:
            return team_name
            
        # Direct lookup
        canonical = self._alias_map.get(team_name.lower())
        if canonical:
            return canonical
            
        # Fuzzy search
        team_lower = team_name.lower()
        for search_term, canonical_names in self._search_index.items():
            if search_term in team_lower:
                if len(canonical_names) == 1:
                    return list(canonical_names)[0]
                    
        # No match found - return original
        logger.debug(f"No canonical name found for: {team_name}")
        return team_name
    
    def get_team_info(self, team_name: str) -> Dict:
        """
        Get comprehensive team information.
        
        Args:
            team_name: Team name (any variant)
            
        Returns:
            Dictionary with canonical_name, short, country, league, aliases, etc.
        """
        canonical = self.standardize(team_name)
        
        if canonical in self.canonical_names:
            info = self.canonical_names[canonical].copy()
            info['canonical_name'] = canonical
            return info
        else:
            return {
                'canonical_name': team_name,
                'short': team_name[:3].upper(),
                'country': 'Unknown',
                'league': 'Unknown',
                'aliases': [],
                'search_terms': []
            }
    
    def search(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for teams matching query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of canonical team names matching query
        """
        query_lower = query.lower()
        matches = set()
        
        # Check canonical names
        for canonical_name in self.canonical_names.keys():
            if query_lower in canonical_name.lower():
                matches.add(canonical_name)
                
        # Check aliases
        for canonical_name, data in self.canonical_names.items():
            for alias in data["aliases"]:
                if query_lower in alias.lower():
                    matches.add(canonical_name)
                    
        # Check search terms
        for search_term, canonical_names in self._search_index.items():
            if query_lower in search_term:
                matches.update(canonical_names)
                
        return sorted(list(matches))[:limit]
    
    def get_all_canonical_names(self) -> List[str]:
        """Get list of all canonical team names."""
        return sorted(self.canonical_names.keys())
    
    def get_teams_by_league(self, league: str) -> List[str]:
        """Get all teams in a specific league."""
        teams = []
        league_lower = league.lower()
        
        for canonical_name, data in self.canonical_names.items():
            if league_lower in data['league'].lower():
                teams.append(canonical_name)
                
        return sorted(teams)
    
    def get_teams_by_country(self, country: str) -> List[str]:
        """Get all teams from a specific country."""
        teams = []
        country_lower = country.lower()
        
        for canonical_name, data in self.canonical_names.items():
            if country_lower in data['country'].lower():
                teams.append(canonical_name)
                
        return sorted(teams)


# Global singleton instance
_team_name_standardizer = None

def get_team_name_standardizer() -> TeamNameStandardizer:
    """Get global team name standardizer instance."""
    global _team_name_standardizer
    if _team_name_standardizer is None:
        _team_name_standardizer = TeamNameStandardizer()
    return _team_name_standardizer


# Convenience functions
def standardize_team_name(team_name: str) -> str:
    """Standardize a team name to canonical form."""
    return get_team_name_standardizer().standardize(team_name)


def search_teams(query: str, limit: int = 10) -> List[str]:
    """Search for teams matching query."""
    return get_team_name_standardizer().search(query, limit)


def get_team_info(team_name: str) -> Dict:
    """Get comprehensive team information."""
    return get_team_name_standardizer().get_team_info(team_name)


# Example usage
if __name__ == "__main__":
    standardizer = get_team_name_standardizer()
    
    print("Team Name Standardization Examples:")
    print("=" * 60)
    
    test_names = [
        "Man City",
        "Barça",
        "Juve",
        "PSG",
        "United",
        "Real",
        "Bayern",
        "Inter"
    ]
    
    for name in test_names:
        canonical = standardizer.standardize(name)
        info = standardizer.get_team_info(name)
        print(f"{name:15} → {canonical:25} ({info['short']}) - {info['league']}")
    
    print("\n" + "=" * 60)
    print("Search Examples:")
    print("=" * 60)
    
    queries = ["manchester", "madrid", "milan"]
    for query in queries:
        results = standardizer.search(query)
        print(f"'{query}' → {', '.join(results)}")
