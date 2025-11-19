#!/usr/bin/env python3
"""
Logo Integration System for GoalDiggers Football Betting Platform
Provides uniform logo display across the platform with fallback options
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

logger = logging.getLogger(__name__)

class LogoSystem:
    """Logo integration system for uniform brand identity"""
    
    def __init__(self):
        self.base_path = self._find_logo_directory()
        self.logo_cache: Dict[str, Image.Image] = {}
        self.loaded_successfully = self._init_logo_cache()
    
    def _find_logo_directory(self) -> Path:
        """Find the correct logo directory with fallbacks"""
        potential_paths = [
            Path(os.path.dirname(os.path.abspath(__file__))) / "../../static/images/logos",
            Path(os.path.dirname(os.path.abspath(__file__))) / "../static/images/logos",
            Path(os.path.dirname(os.path.abspath(__file__))) / "../../static/images",
            Path(os.path.dirname(os.path.abspath(__file__))) / "../static/images",
            Path(os.path.dirname(os.path.abspath(__file__))) / "../../dashboard/static/images",
            Path(os.path.dirname(os.path.abspath(__file__))) / "../dashboard/static/images",
        ]
        
        for path in potential_paths:
            if path.exists():
                logger.info(f"Found logo directory at {path}")
                return path
        
        # Default fallback
        default_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../static/images"
        logger.warning(f"Logo directory not found, using default: {default_path}")
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path
    
    def _init_logo_cache(self) -> bool:
        """Initialize the logo cache with available logos"""
        try:
            # Try to load the main logo
            main_logo_path = self.base_path / "GoalDiggers_logo.png"
            if main_logo_path.exists():
                self.logo_cache["main"] = Image.open(main_logo_path)
            else:
                # Create a fallback text-based logo
                self._create_fallback_logo("main", "GoalDiggers")
            
            # Success if we have at least the main logo
            return "main" in self.logo_cache
            
        except Exception as e:
            logger.error(f"Failed to initialize logo cache: {e}")
            return False
    
    def _create_fallback_logo(self, key: str, text: str) -> None:
        """Create a fallback text-based logo"""
        try:
            # Create a simple text-based logo
            img = Image.new('RGBA', (300, 100), color=(255, 255, 255, 0))
            d = ImageDraw.Draw(img)
            
            # Use a system font as fallback
            try:
                font = ImageFont.truetype("Arial", 36)
            except OSError:
                font = ImageFont.load_default()
            
            # Draw text with brand colors
            d.text((10, 10), text, fill=(0, 123, 255), font=font)
            self.logo_cache[key] = img
            
        except Exception as e:
            logger.error(f"Failed to create fallback logo: {e}")
    
    def render_main_logo(self, width: Optional[int] = None) -> None:
        """Render the main GoalDiggers logo"""
        if "main" in self.logo_cache:
            # Get the logo
            logo = self.logo_cache["main"]
            
            # Prepare for display
            st.image(logo, width=width, caption="")
        else:
            # Text fallback
            st.markdown("# ⚽ GoalDiggers")
    
    def get_team_logo(self, team_name: str) -> Optional[Image.Image]:
        """Get a team logo by name with fallback"""
        # Normalize team name
        key = team_name.lower().replace(" ", "_")
        
        # Return from cache if available
        if key in self.logo_cache:
            return self.logo_cache[key]
        
        # Try to load from disk
        try:
            potential_paths = [
                self.base_path / f"{key}.png",
                self.base_path / f"{key}.jpg",
                self.base_path / "teams" / f"{key}.png",
                self.base_path / "teams" / f"{key}.jpg"
            ]
            
            for path in potential_paths:
                if path.exists():
                    self.logo_cache[key] = Image.open(path)
                    return self.logo_cache[key]
            
            # Create fallback
            self._create_fallback_logo(key, team_name[:2].upper())
            return self.logo_cache[key]
            
        except Exception as e:
            logger.warning(f"Failed to load team logo for {team_name}: {e}")
            return None
    
    def render_team_logo(self, team_name: str, width: int = 50) -> None:
        """Render a team logo by name"""
        logo = self.get_team_logo(team_name)
        if logo:
            st.image(logo, width=width, caption="")
        else:
            st.markdown(f"**{team_name[:3]}**")

# Initialize the logo system
logo_system = LogoSystem()
