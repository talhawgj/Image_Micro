
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import tempfile
import logging

logger = logging.getLogger(__name__)

def get_chrome_driver():
    """
    Initializes a single-use Headless Chrome instance optimized for AWS Lambda.
    No pooling is required because Lambda handles request concurrency inherently.
    """
    logger.info("Starting Lambda-optimized ChromeDriver...")
    options = Options()
    options.binary_location = "/usr/bin/chrome"
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--single-process")
    options.add_argument("--no-zygote")
    options.add_argument("--window-size=1600,1200")
    
    user_data_dir = tempfile.mkdtemp(dir="/tmp")
    data_path = tempfile.mkdtemp(dir="/tmp")
    disk_cache_dir = tempfile.mkdtemp(dir="/tmp")
    
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument(f"--data-path={data_path}")
    options.add_argument(f"--disk-cache-dir={disk_cache_dir}")
    
    try:
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        logger.error(f"Failed to start ChromeDriver: {e}")
        raise RuntimeError(f"ChromeDriver initialization failed: {e}")