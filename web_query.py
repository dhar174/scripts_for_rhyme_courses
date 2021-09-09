
import nltk
import numpy as np
import random
import string # to process standard python strings
from nltk.chat.util import Chat, reflections


from io import StringIO
import sys
import json




import requests
from requests import get
from bs4 import BeautifulSoup
from lxml import html



usr_agent = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/87.0.4280.141 Safari/537.36'}
def fetch_results(search_term, number_results, language_code):
    #First we have to replace spaces in the search query with plus symbols.
    escaped_search_term = search_term.replace(' ', '+')
    #Plug in the search term along with the number of results and the language code.
    google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results+1,
                                                                          language_code)
    #Get the response using requests.get
    response = get(google_url, headers=usr_agent)
    # raise_for_status() returns an HTTPError object if an error has occurred during the process. 
    response.raise_for_status()
    #return the response as text
    return response.text
def parse_results(raw_html):
    #Beautiful Soup is a Python library for pulling data out of HTML and XML files. 
    soup = BeautifulSoup(raw_html, 'html.parser')
    #This will help us find the relevant html attribute
    result_block = soup.find_all('div', attrs={'class': 'g'})
    for result in result_block:
        link = result.find('a', href=True)
        title = result.find('h3')
        if link and title:
            yield link['href']
                
def search(term, num_results=10, lang="en"):
    #This will first call fetch_results above and then use that to return the list of results as a list of URLs
    html = fetch_results(term, num_results, lang)
    return list(parse_results(html))







def chatbot_query(query, index=0):
    #Fallback is just in case of error
    fallback = 'Sorry, I cannot find information regarding that.'
    result = ''

    #This is where Search is called, passing the search query.
    search_result_list = list(search(query, 10))
    #We'll use several try/except blocks to try extracting information from the response
    try:
        #Grab the page
        page = get(search_result_list[index], headers=usr_agent)

        #Prepare for soup
        tree = html.fromstring(page.content)
        #Declare an instance of BeautifulSoup
        soup = BeautifulSoup(page.content, features="lxml")

        article_text = ''
        #Use our soup object to search for HTML <p> element, which indicates a paragraph of text.
        article = soup.findAll('p')
        #For each block of text found, join them in list form
        for element in article:
            article_text += '\n' + ''.join(element.findAll(text = True))
        #Fix the newline characters left over
        article_text = article_text.replace('\n', '')
        #Establish "first_sentence" as a first-guess of an answer
        first_sentence = article_text.split('.')[0]
        
        max_matches = 0
        answers = []
        #Here we are splitting the query into words and checking for the sentence in the response that contains words from the query (to avoid irrelevant text).
        #For each sentence in the article, split by periods, will undergo the following process.
        for sentence in article_text.split('.'):
            num_matches = 0
            #Iterate through each word in the query and compare to the sentence. This is a VERY simple solution.
            for word in query.split(' '):
                if word.lower() in sentence.lower():
                    #Only compare words more than 3 letters long
                    if len(word)>3:
                        num_matches+=1
            #If the number of word matches is zero, we store the sentence for later, putting it in the top position if it has more matches than the previous zero slot,
            #or appending to the bottom of the list otherwise.
            if(num_matches > 0):
                ans = sentence
                if num_matches > max_matches:
                    max_matches = num_matches
                    answers.insert(0,ans)
                else:
                    answers.append(ans)
            #Be sure to check for null values!
                    #If the above process led to even a single result with a match, we will take that as our answer in "first_sentence". Otherwise, we stick to the original, first sentence.
            if(answers !=None and answers[0]!=''):
                
                first_sentence = str(answers[0])
            else:
                if(article_text!=None and article_text!= ''):
                    first_sentence = article_text.split('.')
                else:
                    first_sentence = fallback

    #If there is an error in the above process, the following will try some workarounds for a couple small, known issues with parsing data from Google.
    #It only varies slightly from the above, as a backup in case of error.
    except:
        #Call Google again for the search results
        escaped_search_term = query.replace(' ', '+')
        google_url = 'https://www.google.com/search?q=+define+{}'.format(escaped_search_term)
        reply = get(google_url, headers=usr_agent)
        reply.raise_for_status()

        
        search_result_list = list(parse_results(reply.text))
        try:
            page = get(search_result_list[index], headers=usr_agent)
        except:
            page = get('https://www.google.com'+search_result_list[index], headers=usr_agent)
        
        
        tree = html.fromstring(page.content)

       
        try:
            soup = BeautifulSoup(page.text, "lxml")
            article = soup.find('div', class_='Z0LcW')
            print('article',article.text)
        except:
            soup = BeautifulSoup(page.content, "lxml")
            article = soup.findAll('p')
            

        
        article_text = ''
        
        for element in article:
            article_text += '\n' + ''.join(element.findAll(text = True))
        article_text = article_text.replace('\n', '')
        max_matches = 0
        answers = []
        
        first_sentence = article_text.split('.')
        for sentence in article_text.split('.'):
##            print(sentence)
            num_matches = 0
            for word in query.split(' '):
                if word.lower() in sentence.lower():
                    if(len(word)>=3):
                        num_matches+=1
##                        print(word+sentence)
            if num_matches > 0:
                ans = sentence
                if(num_matches > max_matches and ans != ''):
                    max_matches = num_matches
##                    print("win",ans+' '+str(num_matches))
                    answers.insert(0,ans)
                else:
                    answers.append(ans)
                
        if(answers != None and answers[0] != ''):
            first_sentence = str(answers[0])
##            print(first_sentence)
        else:
            if(article_text!=None and article_text!= ''):
                first_sentence = article_text.split('.')
            else: first_sentence = fallback
    
    chars_without_whitespace = first_sentence.translate(
        { ord(c): None for c in string.whitespace }
    )

    if len(chars_without_whitespace) > 0:
        result = first_sentence
    else:
        result = fallback

    return result
