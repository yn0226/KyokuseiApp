# -*- coding: utf-8 -*-

#必要なモジュール
#-----------------------------
#辞書インストール
#pip install unidic-lite
#↑がだめで、↓でインストールしてmecab使える
#python -m unidic download
#-----------------------------
#import sys
#sys.path.append('C:/Users/2nb23/anaconda3/Lib/site-packages')
#path_list = sys.path
#print(path_list)

import numpy as np
import pandas as pd

#import MeCab
#import oseti

from janome.tokenizer import Tokenizer
import re
import codecs

import csv
#------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
#import torchsummary
#from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger

#-------------------
# pythonファイル参照
#-------------------
from kakakuCom import Scrape, scrape_kakaku    # kakakuCom.py からスクレイピング定義を読み込み
#from NetClass import Net # NetClass.py から前処理とネットワークの定義を読み込み

from flask import Flask, request, render_template, redirect
import io
import base64
from wtforms import Form, StringField, validators, SubmitField

# # #------------------
# # # スクレイピング
# # #------------------
# # # kakakuCom.pyから関数の呼び出し

# # # スクレイピング処理
# res_URL = scrape_kakaku('https://review.kakaku.com/review/M0000000975/#tab')

# # データフレームの表示
# # kakakuCom.py から読み込み
# res_df = res_URL
# print(res_df[:3])
# print('review数:', len(res_df))


#===========================================================================

#------------------
#感情分析クラス
#------------------
class SentimentAnalysis:
    def __init__(self,dic_path):
        """
        コンストラクタ
        """
        self.words = []
        self.dic = self.read_dic(dic_path)
    
    def analyze(self):
        '''
        感情分析

        Returns:
        --------
            (int,int,int,int) 　ポジティブ数、ネガティブ数、ニュートラル数、判定不可数
        '''
        posi = 0
        nega = 0
        neut = 0
        err = 0

        for word in self.words:
            res = self.dic.get(word,'-')            
            if res == 'p':
                posi += 1
            elif res == 'n':
                nega += 1
            elif res == 'e':
                neut += 1
            else:
                err += 1       
        return posi,nega,neut,err 
        

            
                
    def word_separation(self,text):
        """
        形態素解析により名詞、形容詞、動詞を抽出

        ---------------
        Parameters:
            text : str         テキスト
        """
        token = Tokenizer().tokenize(text)
        words = []
    
        for line in token:
            tkn = re.split('\t|,', str(line))
            # 名詞、形容詞、動詞で判定
            if tkn[1] in ['名詞','形容詞','動詞'] :
                words.append(tkn[0])  
        return words

    def read_dic(self,filename):
        with codecs.open(filename,'r','utf-8','ignore') as f:
            lines = f.readlines()    
            dic = { x.split('\t')[0]:x.split('\t')[1] for x in lines }
        return dic

    def read_file(self,filename,encoding='utf-8'):
        '''
        ファイルの読み込み

        Parameters:
        --------
            filename : str  分析対象のファイル名 
        '''
        with codecs.open(filename,'r',encoding,'ignore') as f:
            self.read_text(f.read())

    def read_text(self,text):
        '''
        テキストの読み込み

        Parameters:
        --------
            text : str   分析対象のテキスト
        '''
        # 形態素解析を用いて名詞のリストを作成
        self.words = self.word_separation(text)
#===========================================================================



#------------------
# インスタンスの生成
#------------------
# ★ローカルは以下のパス
# sa = SentimentAnalysis('./kyokuseiDic/pn.csv.m3.120408.trim')
# ★デプロイ時は以下のパス
sa = SentimentAnalysis('../kyokuseiDic/pn.csv.m3.120408.trim')

# Flask のインスタンスを作成
app = Flask(__name__)   

# WTForms を使い、index.html 側で表示させるフォームを構築します。
class InputForm(Form):
    InputFormTest = StringField('価格コムのレビューページのURLを入力してください',
                    [validators.InputRequired()])

    # HTML 側で表示する submit ボタンの表示
    submit = SubmitField('送信')



@app.route('/', methods=['GET', 'POST'])
def predicts():
    # WTFormsで構築したフォームをインスタンス化
    form = InputForm(request.form)

    # POST---
    # Web ページ内の機能に従って処理を行なう    
    if request.method == 'POST':
        
        # 条件に当てはまる場合
        if form.validate() == False:
            return render_template('index.html', forms=form)
            
        
        # 条件に当てはまらない場合:推論を実行
        else:

            #------------------
            # スクレイピング
            #------------------
            # URLチェック
            input_URL = request.form['InputFormTest']

            # URLからスクレイピング(kakakuCom.pyから関数の呼び出し)         
            res_URL = scrape_kakaku(input_URL)

            # データフレームの表示(kakakuCom.py から読み込み)
            res_df = res_URL            

            
            #------------------
            # スクレイピング結果 から読み込み
            #------------------
            df_Input = res_df

            # CSVからレビュー部分を順に取り出し、reviewsに格納
            reviews = []
            for review in df_Input['comment']:    
                reviews.append(review)
                #print('全レビュー数：', len(reviews))

            # reviewsからレビューを順に取り出し、linesに格納
            # →linesから一文ずつ取り出し、感情分析
            lines = []
            result = pd.DataFrame(columns=['結果', 'P', 'N', '中立', 'エラー'])
            i = 0   #結果格納用DFのカウンタ

            all_P = 0
            all_N = 0
            all_Neu = 0
            all_Ex = 0

            for review in reviews:    

                #読み込んだレビューを一文ずつ区切る
                lines = review.split("。")
                for line in lines:
                    if line.strip() != "":
                        line.strip()        #各文の前後の余分なスペースを削除
                        #print('区切った文章の数：',len(line))
                
                #1行ずつ読んで分析する
                cnt_P = 0
                cnt_N = 0
                cnt_Neu = 0
                cnt_Err = 0
                for line in lines:
                    #print(line)         # 元の文書を表示
                    sa.read_text(line)  # 文書の読み込み
                    #res = sa.analyze()  # 感情分析の実行★★
                    posi, nega, neut, err = sa.analyze()  # 感情分析の実行★★

                    #結果をカウントアップ
                    cnt_P = cnt_P + posi
                    cnt_N = cnt_N + nega
                    cnt_Neu = cnt_Neu + neut
                    cnt_Err = cnt_Err + err

                # 1レビューの判定結果の合算から、P/N/中立を決定
                res_PN =''
                if cnt_P > cnt_N:    
                    res_PN = 'P'
                elif cnt_P < cnt_N:
                    res_PN = 'N'
                elif cnt_P == cnt_N:
                    if cnt_P == 0 and cnt_N == 0 and cnt_Neu ==0:
                        res_PN = 'Err'
                    else:
                        if cnt_P == cnt_N:
                            res_PN = 'Neu'
                        elif cnt_P > cnt_Neu:
                            res_PN = 'P'                
                        elif cnt_P < cnt_Neu:
                            res_PN = 'Neu'
                        elif cnt_P == cnt_Neu:
                            res_PN = 'Neu'


                # 結果を総計用カウンタに格納
                if res_PN == 'P':
                    all_P = all_P + 1
                elif res_PN == 'N':
                    all_N = all_N + 1
                elif res_PN == 'Err':
                    all_Ex = all_Ex + 1
                else:
                    all_Neu = all_Neu + 1   
                

                # 値を1行ずつ追加
                result.loc[i] = [res_PN, cnt_P, cnt_N, cnt_Neu, cnt_Err]  # 値を追加
                i = i+1
                #print('No.:',i,'判定結果：',res_PN, cnt_P, cnt_N, cnt_Neu, cnt_Err)
                
            # 新しくDFを作成し、元のCSVデータに判定結果を追加
            #df_CSVRes = pd.concat([df_Input, result], axis=1)

            # インデックスをリセット
            df_Input.reset_index(drop=True, inplace=True)
            result.reset_index(drop=True, inplace=True)

            # 新しくDFを作成し、元のCSVデータに判定結果を追加
            # resultの1列目を抽出
            result_column = result.iloc[:, 0]
            df_CSVRes = pd.concat([df_Input, result_column], axis=1)

            # 総合判定
            all_Res = ''
            if all_P > all_N:
                all_Res = 'ポジティブ'
            elif all_P < all_N:
                all_Res = 'ネガティブ'
            elif all_P == all_N:
                if all_P > all_Neu:
                    all_Res = 'ポジティブ'
                elif all_P < all_Neu:
                    all_Res = '中立'
                elif all_P == all_Neu:
                    all_Res = '中立'
            print('総合判定：',all_Res)

            # ↑のDFをCSVに出力
            #df_CSVRes.to_csv('Result_価格コム.csv', index=None, lineterminator='\n', encoding='shift-jis')
           

            # # dfに結果を追加
            # df_Input['result'] = y3_value            
            # #print(df_Input[:2])

            # 判定結果を送る
            print('[4]総合判定：',all_Res,'P:',all_P,'N:',all_N,'Neu:',all_Neu)            
            return render_template('result.html', Res_NP=all_Res, Res_P=all_P,Res_N=all_N,Res_Neu=all_Neu, table=(df_CSVRes.to_html(classes="mystyle")))
        return redirect(request.url)
    
    # GET ---
    # URL から Web ページへのアクセスがあった時の挙動
    elif request.method == 'GET':
        return render_template('index.html',forms=form)
    
#アプリ実行の定義
if __name__ == '__main__':
    app.run(debug=True)