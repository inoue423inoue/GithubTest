
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://maidragon.jp/1st/character/#/kobayashi"
#download_url = "https://maidragon.jp/1st/character/" なくてよかった

options = Options()
# ヘッドレスモードで実行する場合
# options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

try:
    # 取得先URLにアクセス
    driver.get(url)
    # コンテンツが描画されるまで待機
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, "html.parser")    

finally:
 # プラウザを閉じる
 driver.quit()

img_source = soup.select("img")
for img in img_source :
    src = img.get("src")
    print(src)
    target = urljoin(url, src)
    resp = requests.get(target)
    file = open('picture/test/' + target.split('/')[-1], 'wb')
    file.write(resp.content)
    file.close()