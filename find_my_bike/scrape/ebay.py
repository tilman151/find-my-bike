import logging
import warnings
from datetime import datetime, timedelta
from time import sleep
from typing import Optional, Dict, List

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import (
    staleness_of,
)
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
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
        num: int = 100,
    ) -> List[Dict[str, str]]:
        self._open_website()
        self._search(query, location, category)
        items = self._get_items(num, query, location)
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
        WebDriverWait(self.driver, 10).until(
            staleness_of(self.driver.find_element(By.TAG_NAME, "html"))
        )

    def _search(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        logger.info(f"Search '{query}' in '{location}'")
        if category is not None:
            if category in (categories := self._get_categories()):
                categories[category].click()
            else:
                raise ValueError(f"Category {category} not found")
        if query is not None:
            search_bar = self.driver.find_element(By.ID, "site-search-query")
            search_bar.send_keys(query)
        if location is not None:
            location_bar = self.driver.find_element(By.ID, "site-search-area")
            location_bar.send_keys(location)
        self.driver.find_element(By.ID, "site-search-submit").click()

    def _get_categories(self):
        categories = self.driver.find_elements(By.CLASS_NAME, "text-link-secondary")
        categories = {cat.text: cat for cat in categories}

        return categories

    def _get_items(
        self, num: int, query: str, location: Optional[str] = None
    ) -> List[Dict[str, str]]:
        logger.info(f"Retrieve {num} items")
        items = []
        while len(items) < num:
            items.extend(self._get_items_from_page(query, location))
            try:
                self._navigate_next_result_page()
            except NoSuchElementException:
                break

        return items[:num]

    def _get_items_from_page(
        self, query: str, location: Optional[str] = None
    ) -> List[Dict[str, str]]:
        ads = self.driver.find_elements(By.CLASS_NAME, "aditem")
        logger.debug(f"Get {len(ads)} items from page")
        items = [
            item
            for ad in ads
            if (item := self._get_item(ad, query, location)) is not None
        ]

        return items

    def _get_item(self, ad, query: str, location: str) -> Optional[Dict[str, str]]:
        image = ad.find_element(By.CLASS_NAME, "imagebox")
        image_url = image.get_attribute("data-imgsrc")
        title_tag = ad.find_element(
            By.XPATH, "//h2[contains(@class, 'text-module-begin')]/a"
        )
        url = title_tag.get_attribute("href")
        date = self._wait_until_populated(
            ad.find_element(By.CLASS_NAME, "aditem-main--top--right"),
            raise_error=False,
        )
        if image_url is not None and date is not None:
            item = {
                "image_url": image_url,
                "url": url,
                "query": query,
                "loc_query": location,
                "title": self._wait_until_populated(title_tag),
                "location": self._wait_until_populated(
                    ad.find_element(By.CLASS_NAME, "aditem-main--top--left")
                ),
                "date": self._parse_date(date),
            }
        else:
            logger.debug(f"Skip '{url}' without image")
            item = None

        return item

    def _close_details_tab(self):
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])

    def _open_details_tab(self, url):
        self.driver.execute_script("window.open('about:blank', 'details');")
        self.driver.switch_to.window("details")
        self.driver.get(url)

    def _navigate_next_result_page(self) -> None:
        logger.debug("Navigate to next page")
        next_page_btn = self.driver.find_element(
            By.XPATH,
            "//span[@class='pagination-current']/following-sibling::a[1]",
        )
        next_page_btn.click()

    @staticmethod
    def _parse_date(date: str) -> datetime:
        if "Heute" in date:
            parsed = datetime.today().date()
        elif "Gestern" in date:
            parsed = datetime.today().date() - timedelta(days=1)
        else:
            parsed = datetime.strptime(date, "%d.%m.%Y")

        return parsed

    @staticmethod
    def _wait_until_populated(
        element,
        timeout: float = 1,
        increment: float = 0.1,
        raise_error: bool = True,
    ) -> str:
        passed_time = 0
        while not element.text and passed_time < timeout:
            sleep(increment)
            passed_time += increment

        if not element.text:
            if raise_error:
                raise RuntimeError(f"Timeout while waiting for {element} to populate")
            else:
                text = None
        else:
            text = element.text

        return text
