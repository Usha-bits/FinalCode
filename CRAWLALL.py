import scrapy
import os
import json
import logging
import re
from urllib.parse import urljoin


class GitLabSpider(scrapy.Spider):
    name = "gitlab_handbook"
    allowed_domains = ["handbook.gitlab.com"]
    start_urls = ["https://handbook.gitlab.com/handbook/people-group/"]

    # Define output folders and log file paths
    OUTPUT_FOLDER = "output/ScrapData"
    LOG_FOLDER = "log"
    TEXT_FILE = os.path.join(OUTPUT_FOLDER, "text_data.json")
    TABLE_FILE = os.path.join(OUTPUT_FOLDER, "table_data.json")
    LOG_FILE = os.path.join(LOG_FOLDER, "scrapdata_log.txt")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    custom_settings = {
        "DEPTH_LIMIT": 2,
        "REDIRECT_ENABLED": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 2,
        "AUTOTHROTTLE_MAX_DELAY": 10,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "LOG_FILE": LOG_FILE,
        "LOG_LEVEL": "INFO",
    }

    def parse(self, response):
        logging.info(f"Scraping: {response.url}")
        title = response.xpath("//title/text()").get()

        # Extract content
        text_data = self.extract_text_data(response)
        table_data = self.extract_table_data(response)

        # Save as JSONL (one line per record)
        self.save_data(self.TEXT_FILE, {
            "url": response.url,
            "title": title,
            "content": text_data
        })
        self.save_data(self.TABLE_FILE, {
            "url": response.url,
            "title": title,
            "content": table_data
        })

        # Crawl internal links
        self.follow_internal_links(response)

    def extract_text_data(self, response):
        text_data = {}
        current_heading = "Introduction"

        for element in response.xpath("//h1 | //h2 | //h3 | //p | //ul | //ol | //li | //div"):
            tag = element.root.tag
            content = element.xpath("string()").get(default="").strip()

            if not content:
                continue

            if tag in ["h1", "h2", "h3"]:
                current_heading = content
                if current_heading not in text_data:
                    text_data[current_heading] = ""
            else:
                text_data.setdefault(current_heading, "")
                text_data[current_heading] += " " + content

        for key in text_data:
            text_data[key] = re.sub(r"\s+", " ", text_data[key]).strip()

        return text_data

    def extract_table_data(self, response):
        table_data = {}
        current_heading = "Introduction"

        for element in response.xpath("//h1 | //h2 | //h3 | //table"):
            tag = element.root.tag
            if tag in ["h1", "h2", "h3"]:
                current_heading = element.xpath("string()").get().strip()
            elif tag == "table":
                rows = self.extract_table_rows(element)
                if rows:
                    table_data.setdefault(current_heading, []).append(rows)

        return table_data

    def extract_table_rows(self, table_element):
        table_rows = []
        for row in table_element.xpath(".//tr"):
            cells = row.xpath(".//th//text() | .//td//text()").getall()
            cleaned = [c.strip() for c in cells if c.strip()]
            if cleaned:
                table_rows.append(cleaned)
        return table_rows

    def save_data(self, file_path, data):
        """Append structured data as a single JSON line."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + "\n")
        except Exception as e:
            logging.error(f" Error saving JSON to {file_path}: {e}")

    def follow_internal_links(self, response):
        for href in response.xpath("//a[@href]/@href").getall():
            full_url = urljoin(response.url, href)
            if full_url.startswith("https://handbook.gitlab.com/handbook/people-group/"):
                yield scrapy.Request(url=full_url, callback=self.parse)
