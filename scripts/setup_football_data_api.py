#!/usr/bin/env python
"""
Football-Data.org API Integration Setup Script

This script helps set up the API key for the Football-Data.org API and validates
that everything is working correctly. It will:

1. Check if a .env file exists and create one from template if not
2. Prompt the user to input their API key if needed
3. Verify that the API key works
4. Update the configuration to use the API key

Usage:
    python setup_football_data_api.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api_setup")

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    # Import the dotenv module for .env file handling
    from dotenv import load_dotenv, set_key
    dotenv_available = True
except ImportError:
    logger.warning("python-dotenv module not installed. Install with: pip install python-dotenv")
    dotenv_available = False

# Try to import the HTTP client for API testing
try:
    from utils.http_client_async import HttpClientAsync
    http_client_available = True
except ImportError:
    logger.warning("HTTP client module not available. API validation will be skipped.")
    http_client_available = False

def check_dotenv_file():
    """Check if .env file exists and create from template if not."""
    env_path = Path(project_root) / ".env"
    env_template_path = Path(project_root) / ".env.template"
    
    if env_path.exists():
        logger.info(".env file found")
        return True
    
    if env_template_path.exists():
        try:
            # Create .env from template
            with open(env_template_path, "r") as template_file:
                template_content = template_file.read()
            
            with open(env_path, "w") as env_file:
                env_file.write(template_content)
                
            logger.info(".env file created from template")
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False
    else:
        logger.error(".env.template file not found")
        return False

def get_api_key_from_user():
    """Prompt the user to input their API key."""
    print("\nPlease enter your Football-Data.org API key.")
    print("If you don't have one, get a free key at: https://www.football-data.org/client/register")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        logger.warning("No API key provided.")
        return None
    
    return api_key

def save_api_key_to_dotenv(api_key):
    """Save the API key to the .env file."""
    if not dotenv_available:
        logger.warning("python-dotenv not available, can't save API key to .env file")
        logger.info(f"Please manually add your API key to the .env file as FOOTBALL_DATA_API_KEY={api_key}")
        return False
        
    env_path = Path(project_root) / ".env"
    try:
        set_key(str(env_path), "FOOTBALL_DATA_API_KEY", api_key)
        logger.info("API key saved to .env file")
        
        # Also reload the environment to make the key available
        load_dotenv(env_path, override=True)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save API key to .env file: {e}")
        return False

async def validate_api_key(api_key):
    """Validate that the API key works."""
    if not http_client_available:
        logger.warning("HTTP client not available, skipping API key validation")
        return None
        
    logger.info("Validating API key...")
    
    client = HttpClientAsync("https://api.football-data.org/v4")
    try:
        response = await client.get(
            "/competitions",
            headers={"X-Auth-Token": api_key, "Accept": "application/json"}
        )
        
        if response.status == 200:
            data = await response.json()
            competition_count = len(data.get("competitions", []))
            logger.info(f"✅ API key is valid! Found {competition_count} competitions")
            return True
        elif response.status == 401:
            logger.error("❌ API key is invalid (Unauthorized)")
            return False
        else:
            logger.error(f"❌ Unexpected response: HTTP {response.status}")
            try:
                error_text = await response.text()
                logger.error(f"Error response: {error_text}")
            except:
                pass
            return False
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return False
    finally:
        await client.close()

async def setup_api():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("⚽ Football-Data.org API Setup")
    logger.info("=" * 60)
    
    # Check for .env file
    if not check_dotenv_file():
        logger.error("Failed to set up .env file. Please create one manually.")
        return False
    
    # Load environment variables
    if dotenv_available:
        load_dotenv()
    
    # Check for existing API key
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        api_key = os.environ.get("FOOTBALL_DATA_TOKEN")
    
    if api_key:
        logger.info("API key found in environment variables")
        valid = await validate_api_key(api_key)
        
        if valid is True:
            logger.info("Existing API key is valid!")
            return True
        elif valid is False:
            logger.warning("Existing API key is invalid. Let's set a new one.")
            api_key = None
        # If valid is None, validation was skipped
    
    # If no key or invalid key, get from user
    if not api_key:
        api_key = get_api_key_from_user()
        
        if not api_key:
            logger.error("No API key provided. Setup cannot continue.")
            return False
            
        # Validate new key
        valid = await validate_api_key(api_key)
        
        if valid is False:
            logger.error("The provided API key is invalid.")
            return False
            
        # Save the key
        if not save_api_key_to_dotenv(api_key):
            logger.warning("Could not automatically save API key.")
            print(f"\nPlease manually add your API key to the .env file:")
            print(f"FOOTBALL_DATA_API_KEY={api_key}")
    
    logger.info("=" * 60)
    logger.info("✅ Football-Data.org API setup complete!")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(setup_api())
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
