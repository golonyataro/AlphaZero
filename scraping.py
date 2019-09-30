import requests
from bs4 import BeautifulSoup
 
r = requests.get("https://ja.wikipedia.org/wiki/日本の女優一覧")
 
soup = BeautifulSoup(r.content, "html.parser")
 
for a in soup.select("li > a"):
  if a.get('title'):
    print(a.get('title'))