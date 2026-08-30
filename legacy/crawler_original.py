# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 08:47:19 2021

@author: Hamid.Jahani
"""

#Crawl The Site
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

def get_data():
    Data=pd.DataFrame()
    baseurl = "https://kalatik.com/21-mobile#/%D9%85%D9%88%D8%AC%D9%88%D8%AF_%D8%A8%D9%88%D8%AF%D9%86-%D9%85%D9%88%D8%AC%D9%88%D8%AF%DB%8C"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
    productlinks = []
    t={}
    data=[]
    c=0
    for x in range(1,8):
        k = requests.get('https://kalatik.com/21-mobile#/%D9%85%D9%88%D8%AC%D9%88%D8%AF_%D8%A8%D9%88%D8%AF%D9%86-%D9%85%D9%88%D8%AC%D9%88%D8%AF%DB%8C/page-{}'.format(x)).text
        soup=BeautifulSoup(k,'html.parser')
        productlist = soup.find_all("a",{"class":"product_img_link"})
        
        for product in productlist:
            link = product.get('href')
            productlinks.append(link)
    
    
    for link in productlinks:
        f = requests.get(link,headers=headers).text
        hun=BeautifulSoup(f,'html.parser')
    
        try:
            x=hun.find("div",{"class":"rte align_justify"})
            name=x.find(["h2"]).text.replace('\n',"")
        except:
            name = None    
        
        try:
            price_discrite=[s for s in hun.find("p",{"class":"our_price_display"}).text.replace('\n',"") if s.isdigit()]
            price=price_discrite[0]
            for s in price_discrite[1:]:
                price=price+s
            price=int(price)
        except:
            price = None
    
        try:
            feature_name=[]
            feature_value=[]
            x=hun.find("div",{"role":"tabpanel"})
            y=x.find_all("div",{"class":"features"})
            for i in y:
                feature_name.append(i.find("span",{"class":"feature_name"}).text.replace('\n',""))
            for i in y:
                feature_value.append(i.find("span",{"class":"feature_value"}).text.replace('\n',""))
            result = dict(zip(feature_name, feature_value))
        except:
            result=None
        result["Price"]=price
        result["Product"]=name
    
        Data = Data.append(result, ignore_index=True)
    return Data