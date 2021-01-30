import requests
from time import time, sleep
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PATH = './chromedriver.exe'
options = Options()
options.add_argument('--log-level=3')
options.add_argument('--headless')
driver = webdriver.Chrome(PATH, options=options)
url = 'https://www.coindesk.com/price/dogecoin'

driver.get(url)
driver.find_element_by_class_name('dropdown-header-title').click()
driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()
#print(driver.find_element(By.CLASS_NAME, 'price-large').text)

while True:
    now = datetime.now().strftime('%S')
    if now == '00':
        print(driver.find_element_by_class_name('price-large').text)
    sleep(1)