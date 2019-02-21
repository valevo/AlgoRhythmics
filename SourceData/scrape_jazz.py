from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import shutil
import urllib.request

def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        print(e)
        return None

def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code==200 and content_type is not None and
            content_type.find('html') > -1)

raw = simple_get('https://bushgrafts.com/midi/')
print(len(raw))

jazz = BeautifulSoup(raw, 'html.parser')
for link in jazz.find_all('a'):
    url = link.get('href')
    if not url:
        continue

    if url[-3:] == 'mid' or url[-3:] == 'MID':
        print(url)
        name = url[44:].replace('%20', '_')
        if len(name) == 0:
            print('skipping...')
            continue

        print(name)
        with urllib.request.urlopen(url) as response, open('./jazz_midi/'+name,
                                                           'wb') as out_file:
            shutil.copyfileobj(response, out_file)


