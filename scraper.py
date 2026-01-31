import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

def scrape_all():
    page = 1
    results = []

    while True:
        url = BASE_URL.format(page)
        resp = requests.get(url)
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        products = soup.select("article.product_pod")

        for p in products:
            title = p.select_one("h3 a")["title"]
            price_text = p.select_one(".product_price .price_color").get_text()
            price = float(re.sub(r"[^\d.]", "", price_text))
            stock = p.select_one(".product_price .availability").get_text(strip=True)
            results.append((title, price, stock))

        page += 1

    return results

if __name__ == "__main__":
    data = scrape_all()
    df = pd.DataFrame(data, columns=["title", "price", "availability"])
    print(df.head())
    print(f"\nTotal books: {len(df)}")

    df.to_csv("scraped_books.csv", index=False)
    print("The scraped file is saved as scraped_books.csv")