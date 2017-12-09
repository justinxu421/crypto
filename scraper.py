# Import libraries
import requests
import csv
import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

f = csv.writer(open('new_gdax_bitcoin_data.csv', 'w'))
f.writerow(['Timestamp','Open','High','Low','Close','Volume (BTC)','Volume (Currency)','Weighted Price'])


start_date = datetime.date(2017,1,1)
end_date = datetime.date(2017,11,11)

d = start_date
delta = datetime.timedelta(days=1)

driver = webdriver.Chrome()
while d <= end_date:
    date = d.strftime("%m-%d")
    print(date)
    url = 'https://bitcoincharts.com/charts/coinbaseUSD#rg2zig1-minzczsg2017-' + date + 'zeg2017-' + date + 'ztgSzm1g10zm2g25'
    driver.get(url)
    driver.refresh()

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    link = driver.find_element_by_link_text('Load raw data')
    link.click()
    #click button
    time.sleep(2)

    # Create a BeautifulSoup object
    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")
    #pull the table        
    chart_and_table = soup.body.find(class_='data')
    entries = chart_and_table.tbody.find_all('tr')

    # Create for loop to print entries
    for i, entry in enumerate(entries):
        if(i == 0):
            continue
        data = entry.find_all('td')
        timestamp = data[0].contents[0]
        Open = data[1].contents[0]
        high = data[2].contents[0]
        low = data[3].contents[0]
        close = data[4].contents[0]
        volumeBTC = data[5].contents[0]
        volumeCurrency = data[6].contents[0]
        weightedPrice = data[7].contents[0]
        f.writerow([str(timestamp.encode('utf8'))[2:-1],str(Open.encode('utf8'))[2:-1],str(high.encode('utf8'))[2:-1],\
                    str(low.encode('utf8'))[2:-1],str(close.encode('utf8'))[2:-1],str(volumeBTC.encode('utf8'))[2:-1],\
                    str(volumeCurrency.encode('utf8'))[2:-1],str(weightedPrice.encode('utf8'))[2:-1]])
    d += delta