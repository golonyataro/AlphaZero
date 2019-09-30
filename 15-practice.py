from selenium import webdriver
from pyvirtualdisplay import Display

display = Display(visible=0, size=(800, 600))
display.start()

browser = webdriver.Chrome()
browser.get('google.com')