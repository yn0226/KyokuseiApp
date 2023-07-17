# 必要なモジュール
# スクレイピング-------------------
import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import time
import datetime

# 形態素解析--------------------------------
#辞書インストール
#pip install unidic-lite
#↑がだめで、↓でインストールしてmecab使える
#python -m unidic download

import numpy as np
import pandas as pd

#import MeCab
#import oseti

from janome.tokenizer import Tokenizer
import re
import codecs
import csv


#=============================
# スクレイピング
#=============================

#------------------------
# [Class]スクレイピング
#------------------------
class Scrape():
 
    def __init__(self,wait=1,max=None):
        self.response = None
        self.df = pd.DataFrame()
        self.wait = wait
        self.max = max
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"}
        self.timeout = 5
 
    def request(self,url,wait=None,max=None,console=True):
        '''
        指定したURLからページを取得する。
        取得後にwaitで指定された秒数だけ待機する。
        max が指定された場合、waitが最小値、maxが最大値の間でランダムに待機する。
 
        Params
        ---------------------
        url:str
            URL
        wait:int
            ウェイト秒
        max:int
            ウェイト秒の最大値
        console:bool
            状況をコンソール出力するか
        Returns
        ---------------------
        soup:BeautifulSoupの戻り値
        '''
        self.wait = self.wait if wait is None else wait
        self.max = self.max if max is None else max
 
        start = time.time()     
        response = requests.get(url,headers=self.headers,timeout = self.timeout)
        time.sleep(random.randint(self.wait,self.wait if self.max is None else self.max))
        
        if console:
            tm = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            lap = time.time() - start
            print(f'{tm} : {url}  経過時間 : {lap:.3f} 秒')
 
        return BeautifulSoup(response.content, "html.parser")
      
    def get_href(self,soup,contains = None):
        '''
        soupの中からアンカータグを検索し、空でないurlをリストで返す
        containsが指定された場合、更にその文字列が含まれるurlだけを返す
 
        Params
        ---------------------
        soup:str
            BeautifulSoupの戻り値
        contains:str
            抽出条件となる文字列            
 
        Returns
        ---------------------
        return :[str]
            条件を満たすurlのリスト
        '''
        urls = list(set([url.get('href') for url in soup.find_all('a')]))
        if contains is not None:
           return [url for url in urls if self.contains(url,contains)]
        return [url for url in urls if urls is not None or urls.strip() != '']
 
    def get_src(self,soup,contains = None):
        '''
        soupの中からimgタグを検索し、空でないsrcをリストで返す
        containsが指定された場合、更にその文字列が含まれるurlだけを返す
 
        Params
        ---------------------
        soup:str
            BeautifulSoupの戻り値
        contains:str
            抽出条件となる文字列            
 
        Returns
        ---------------------
        return :[str]
            条件を満たすurlのリスト
        '''
        urls = list(set([url.get('src') for url in soup.find_all('img')]))
        if contains is not None:
           return [url for url in urls if contains(url,self.contains)]
        return [url for url in urls if urls is not None or urls.strip() != '']
 
    def contains(self,line,kwd):
        '''
        line に kwd が含まれているかチェックする。
        line が None か '' の場合、或いは kwd が None 又は '' の場合は Trueを返す。
 
        Params
        ---------------------      
        line:str
            HTMLの文字列
        contains:str
            抽出条件となる文字列            
 
        Returns
        ---------------------
        return :[str]
            条件を満たすurlのリスト
        '''
        if line is None or line.strip() == '':
            return False
        if kwd is None or kwd == '':
            return True
        return kwd in line 
    
       
    def omit_char(self,values,omits):
        '''
        リストで指定した文字、又は文字列を削除する
 
        Params
        ---------------------      
        values:str
            対象文字列
        omits:str
            削除したい文字、又は文字列            
 
        Returns
        ---------------------
        return :str
            不要な文字を削除した文字列
        '''
        for n in range(len(values)):
            for omit in omits:
                values[n] = values[n].replace(omit,'')
        return values
 
    def add_df(self,values,columns,omits = None):
        '''
        指定した値を　DataFrame に行として追加する
        omits に削除したい文字列をリストで指定可能
 
        Params
        ---------------------      
        values:[str]
            列名
        omits:[str]
            削除したい文字、又は文字列            
        '''
        if omits is not None:
            values = self.omit_char(values,omits)
            columns = self.omit_char(columns,omits)
        
        df = pd.DataFrame(values,index=self.rename_column(columns))
        self.df = pd.concat([self.df,df.T])
        
   
    def to_csv(self,filename,dropcolumns=None):
        '''
        DataFrame をCSVとして出力する
        dropcolumns に削除したい列をリストで指定可能
 
        Params
        ---------------------      
        filename:str
            ファイル名
        dropcolumns:[str]
            削除したい列名            
        '''
        if dropcolumns is not None:
            self.df.drop(dropcolumns,axis=1,inplace=True) 
        self.df.to_csv(filename,index=False,encoding="shift-jis",errors="ignore")
    
    def get_text(self,soup):
        '''
        渡された soup が Noneでなければ textプロパティの値を返す
 
        Params
        ---------------------      
        soup: bs4.element.Tag
            bs4でfindした結果の戻り値
          
        Returns
        ---------------------
        return :str
            textプロパティに格納されている文字列
        '''
 
        return ' ' if soup == None else soup.text
    
    def rename_column(self,columns):
        '''
        重複するカラム名の末尾に連番を付与し、ユニークなカラム名にする
            例 ['A','B','B',B'] → ['A','B','B_1','B_2']
 
        Params
        ---------------------      
        columns: [str]
            カラム名のリスト
          
        Returns
        ---------------------
        return :str
            重複するカラム名の末尾に連番が付与されたリスト
        '''
        lst = list(set(columns))
        for column in columns:
            dupl = columns.count(column)
            if dupl > 1:
                cnt = 0
                for n in range(0,len(columns)):
                    if columns[n] == column:
                        if cnt > 0:
                            columns[n] = f'{column}_{cnt}'
                        cnt += 1
        return columns
 
    def write_log(self,filename,message):
        '''
        指定されたファイル名にmessageを追記する。
 
        Params
        ---------------------      
        filename: str
            ファイル名
        message: str
            ファイルに追記する文字列          
        '''
        message += '\n'
        with open(filename, 'a', encoding='shift-jis') as f:
           f.write(message)
           print(message)
 
    def read_log(self,filename):
        '''
        指定されたファイル名を読み込んでリストで返す
 
        Params
        ---------------------      
        filename: str
            ファイル名
           
        Returns
        ---------------------
        return :[str]
            読み込んだ結果
        '''
        with open(filename, 'r', encoding='shift-jis') as f:
           lines = f.read()
        return lines
    
    # dfの表示処理
    def display_df(self):
        return self.df
#-------------------ここでClass終わり-----------------------

#------------------------
# スクレイピング実行
#------------------------
def scrape_kakaku(url):
    scr = Scrape(wait=2,max=5)
 
    #レビューのURLから商品IDの手前までを取り出す
    url = url[:url.find('#tab')]
 
    for n in range(1,1000):
        #商品の指定ページのURLを生成
        target = url+f'?Page={n}#tab'
        print(f'get：{target}')
 
        #レビューページの取得
        soup = scr.request(target)
        #ページ内のレビュー記事を一括取得
        reviews = soup.find_all('div',class_='revMainClmWrap')
        #ページ内のすべてと評価を一括取得
        evals = soup.find_all('div',class_='reviewBoxWtInner')
        
        #print(f'レビュー数:{len(reviews)}')
        
        #ページ内の全てのレビューをループで取り出す
        for review,eval in zip(reviews,evals):
            #レビューのタイトルを取得
            title = scr.get_text(review.find('div',class_='reviewTitle'))
            #レビューの内容を取得
            comment = scr.get_text(review.find('p',class_='revEntryCont')).replace('<br>','')
 
            #満足度（デザイン、処理速度、グラフィック性能、拡張性、・・・・・の値を取得
            tables = eval.find_all('table')
            star = scr.get_text(tables[0].find('td'))
            date = scr.get_text(eval.find('p',class_='entryDate clearfix'))
            date = date[:date.find('日')+1]
            ths = tables[1].find_all('th')
            tds = tables[1].find_all('td')
 
            columns = ['title','star','date','comment']
            values = [title,star,date,comment] 
 
            for th,td in zip(ths,tds):
                columns.append(th.text)
                values.append(td.text)
            
            #DataFrameに登録
            scr.add_df(values,columns,['<br>'])
            
        
        #ページ内のレビュー数が15未満なら、最後のページと判断してループを抜ける
        if len(reviews) < 15:
            break
    #スクレイプ結果をCSVに出力
    #scr.to_csv("C:/01_Amazon/価格com口コミ.csv")
    
    #スクレイプ結果をDFに出力(app.pyから使用)
    df_return = scr.display_df()
    return df_return

 
#------------------------
# スクレイピングURL指定
#------------------------
#URL指定
#scrape_kakaku('https://review.kakaku.com/review/J0000037949/#tab')

#1.HiKOKI：FWH14DGL
#レビュー40件
#scrape_kakaku('https://review.kakaku.com/review/K0000659714/#tab')

#2.ZABOON TW-127XP2L
#レビュー11件
#scrape_kakaku('https://review.kakaku.com/review/J0000039181/#tab')

#3.まっ直ぐドラム AQW-DX12N
#レビュー28件
#scrape_kakaku('https://review.kakaku.com/review/J0000039939/#tab')

#4.HP 15s-eq3000 G3
#レビュー64件
#scrape_kakaku('https://review.kakaku.com/review/K0001454174/#tab')

#6. Dyson Pure Cool Link タワーファン TP03WS
#レビュー38件
#scrape_kakaku('https://review.kakaku.com/review/K0000956013/#tab')

#7. AQUOS sense7
#レビュー112件
#scrape_kakaku('https://review.kakaku.com/review/M0000000976/#tab')

#8. メディクイックH 頭皮のメディカルシャンプー 320ml
#レビュー3件
#scrape_kakaku('https://review.kakaku.com/review/S0000894529/#tab')

#9. ディアボーテ オイルインシャンプー リッチ&リペア 360ml 詰め替え用
#レビュー2件
#scrape_kakaku('https://review.kakaku.com/review/S0000818063/#tab')

#10. エリクシールシュペリエル メーククレンジングオイルN 150ml
#レビュー25件
#scrape_kakaku('https://review.kakaku.com/review/K0000491615/#tab')

#推論用：
#レビュー16件
#scrape_kakaku('https://review.kakaku.com/review/J0000030692/#tab')
