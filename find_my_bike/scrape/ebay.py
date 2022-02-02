import logging
import warnings
from typing import Optional, Dict, List

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait

SITE_URL = "https://www.ebay-kleinanzeigen.de/"

logger = logging.getLogger(__name__)


class EbayImageScraper:
    headless: bool
    driver: webdriver.Chrome

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

        options = webdriver.ChromeOptions()
        options.headless = self.headless
        self.driver = webdriver.Chrome(options=options)

    def __enter__(self) -> "EbayImageScraper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()

    def quit(self):
        self.driver.quit()

    def get_items(
        self, query: str, location: Optional[str] = None, num: int = 100
    ) -> List[Dict[str, str]]:
        self._open_website()
        self._search(query, location)
        items = self._get_items(num)
        if len(items) < num:
            warnings.warn(f"Could only fetch {len(items)} items")

        return items

    def _open_website(self) -> None:
        logger.info(f"Open {SITE_URL}")
        self.driver.get(SITE_URL)
        self._reject_cookies()

    def _reject_cookies(self) -> None:
        logger.info("Reject cookies")
        options_btn = WebDriverWait(self.driver, 10).until(
            lambda d: d.find_element(By.ID, "gdpr-banner-cmp-button")
        )
        options_btn.click()
        continue_btn = WebDriverWait(self.driver, 10).until(
            lambda d: d.find_element(
                By.XPATH, "/html/body/div[1]/div[2]/div/div[1]/button[1]"
            )
        )
        continue_btn.click()

    def _search(self, query: str, location: Optional[str] = None) -> None:
        logger.info(f"Search '{query}' in '{location}'")
        search_bar = self.driver.find_element(By.ID, "site-search-query")
        search_bar.send_keys(query)
        if location is not None:
            location_bar = self.driver.find_element(By.ID, "site-search-area")
            location_bar.send_keys(location)
        search_bar.send_keys(Keys.ENTER)

    def _get_items(self, num) -> List[Dict[str, str]]:
        logger.info(f"Retrieve {num} items")
        items = []
        while len(items) < num:
            items.extend(self._get_items_from_page())
            try:
                self._navigate_next_result_page()
            except NoSuchElementException:
                break

        return items[:num]

    def _get_items_from_page(self) -> List[Dict[str, str]]:
        ads = self.driver.find_elements(By.CLASS_NAME, "aditem")
        logger.debug(f"Get {len(ads)} items from page")
        items = []
        for aditem in ads:
            image = aditem.find_element(By.CLASS_NAME, "imagebox")
            image_url = image.get_attribute("data-imgsrc")
            title_tag = aditem.find_element(By.XPATH, "div[2]/div[2]/h2/a")
            url = title_tag.get_attribute("href")
            if image_url is not None:
                items.append({"image_url": image_url, "url": url})
            else:
                logger.debug(f"Skip '{url}' without image")

        return items

    def _navigate_next_result_page(self) -> None:
        logger.debug("Navigate to next page")
        next_page_btn = self.driver.find_element(
            By.XPATH,
            "//span[@class='pagination-current']/following-sibling::a[1]",
        )
        next_page_btn.click()
