import random
import time

import certifi
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from utils.proxy_manager import (ProxyManager, UserAgentManager,
                                 create_proxy_manager_from_env,
                                 user_agent_manager)

# Initialize proxy manager
proxy_manager = create_proxy_manager_from_env() or ProxyManager(["direct"])


def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.verify = certifi.where()  # Ensure proper certificate verification
    return session

def make_request(url):
    session = create_session()
    # Use user_agent_manager to get a rotating user agent
    headers = {'User-Agent': user_agent_manager.get_next_user_agent()}
    proxy = proxy_manager.get_proxy()
    try:
        response = session.get(url, headers=headers, proxies={'http': proxy, 'https': proxy})
        response.raise_for_status()
        # Rotate the proxy for the next request
        proxy_manager.rotate_proxy()
        time.sleep(random.uniform(1, 3))  # Random delay between requests
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        proxy_manager.mark_proxy_as_failed(proxy)
        proxy_manager.rotate_proxy()
        return None

