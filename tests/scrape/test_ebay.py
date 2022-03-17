from unittest import mock

from find_my_bike.scrape.ebay import EbayImageScraper


@mock.patch("find_my_bike.scrape.ebay.webdriver")
def test_driver_init(mock_webdriver):
    scraper = EbayImageScraper()
    mock_webdriver.Chrome.assert_called_once_with(
        options=mock_webdriver.ChromeOptions()
    )
    assert scraper.driver is mock_webdriver.Chrome()


@mock.patch("find_my_bike.scrape.ebay.webdriver")
def test_context_manager(mock_webdriver):
    scraper = EbayImageScraper()
    with scraper as context_scraper:
        assert scraper is context_scraper
    mock_webdriver.Chrome().quit.assert_called_once()


def test_retrieving_items():
    with EbayImageScraper() as scraper:
        items = scraper.get_items("Fahrrad", "Berlin", "Fahrräder & Zubehör", 30)
    assert len(items) == 30
    assert "url" in items[0]
    assert "image_url" in items[0]
    assert "query" in items[0]
    assert "location" in items[0]


def test_high_res_image_urls():
    with EbayImageScraper(high_res=True) as scraper:
        items = scraper.get_items("Fahrrad", "Berlin", "Fahrräder & Zubehör", 1)
    assert items[0]["image_url"].endswith("$_59.JPG")  # URL for highres image
