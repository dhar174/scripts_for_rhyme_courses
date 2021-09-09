from lxml import html
import requests
import re
import urllib.request
import shutil







url='https://www.gutenberg.org/ebooks/1727'
mainWebsite="https://www.gutenberg.org/"

page = requests.get(url)
tree = html.fromstring(page.content)

buyers = tree.xpath('//a[@title="Download"]/text()')

prices = tree.xpath('//a[@charset="utf-8"]/text()')

if 'Plain Text UTF-8' in prices:
    
    #soup = BeautifulSoup(r, "html.parser")
    print(url)
    r = requests.get(url, allow_redirects=True)
    #open('temp.txt', 'w').write(r.text)
    #print(r.text)
    
    links = re.findall('a href="(.*txt)"', r.text)
    print(links)
    for l in links:
    
        fileurl= mainWebsite+l
        print(fileurl)
##            r2 = requests.get(fileurl, allow_redirects=True)
        with urllib.request.urlopen(fileurl) as response, open('trainingtext.txt', 'wb') as out_file:
            shutil.copyfileobj(response, out_file)##                print(f)
##                html = f.read().decode('utf-8')
##                print(html)
##                open('trainingtext.txt', 'a+').write(html)
            break
            
        
##    cheese = tree.xpath('//a[@class="link"]/text()')
##    print(cheese)





