import requests
from time import time, sleep
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def collect_price_data_btc():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/bitcoin'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            open = float(open)
            high = high.replace(',','')
            high = float(high)
            low = low.replace(',','')
            low = float(low)
            volume = float(volume) * 1000000000
            volume = int(volume)
            price = price.replace(',','')
            price = float(price)
            price_info = {'open':open,'high':high,'low':low,'volume':volume,'price':price}
            data = pd.DataFrame(data = price_info, index=[0])
            return data

def collect_price_data_eth():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/ethereum'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            open = float(open)
            high = high.replace(',','')
            high = float(high)
            low = low.replace(',','')
            low = float(low)
            volume = float(volume) * 1000000000
            volume = int(volume)
            price = price.replace(',','')
            price = float(price)
            price_info = {'open':open,'high':high,'low':low,'volume':volume,'price':price}
            data = pd.DataFrame(data = price_info, index=[0])
            return data
    
def collect_price_data_ltc():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/litecoin'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            open = float(open)
            high = high.replace(',','')
            high = float(high)
            low = low.replace(',','')
            low = float(low)
            volume = float(volume) * 1000000000
            volume = int(volume)
            price = price.replace(',','')
            price = float(price)
            price_info = {'open':open,'high':high,'low':low,'volume':volume,'price':price}
            data = pd.DataFrame(data = price_info, index=[0])
            return data

def collect_price_data_xrp():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/xrp'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            open = float(open)
            high = high.replace(',','')
            high = float(high)
            low = low.replace(',','')
            low = float(low)
            volume = float(volume) * 1000000000
            volume = int(volume)
            price = price.replace(',','')
            price = float(price)
            price_info = {'open':open,'high':high,'low':low,'volume':volume,'price':price}
            data = pd.DataFrame(data = price_info, index=[0])
            return data

def collect_price_data_xlm():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/stellar'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            open = float(open)
            high = high.replace(',','')
            high = float(high)
            low = low.replace(',','')
            low = float(low)
            volume = float(volume) * 1000000000
            volume = int(volume)
            price = price.replace(',','')
            price = float(price)
            price_info = {'open':open,'high':high,'low':low,'volume':volume,'price':price}
            data = pd.DataFrame(data = price_info, index=[0])
            return data