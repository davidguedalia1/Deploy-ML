from sentence_transformers import SentenceTransformer
import numpy as np 
import pandas as pd


class Preprocess:
  def __init__(self, df):
    self.df = df
    self.lst_domain = None
      
  def create_data_to_predict(self):
    df = self.feature_engineer(self.df)
    df_nlp = df[['title', 'body', 'domain']]
    df_ml = df.drop(['title', 'body', 'domain'], axis=1)
    model_transform = SentenceTransformer('all-MiniLM-L6-v2')
    df_nlp_list_title = df_nlp.iloc[:,0].values.tolist()
    df_nlp_encode_title = model_transform.encode(df_nlp_list_title)
    df_nlp_list_body = df_nlp.iloc[:,1].values.tolist()
    df_nlp_encode_body = model_transform.encode(df_nlp_list_body)
    df_nlp_list_domain = df_nlp.iloc[:,2].values.tolist()
    df_nlp_encode_domain = model_transform.encode(df_nlp_list_domain)
    x_merge = np.concatenate((df_nlp_encode_title,df_nlp_encode_body),axis=1)
    x_merge = np.concatenate((x_merge,df_nlp_encode_domain),axis=1)
    x_ml = df_ml.to_numpy()
    x_final = np.concatenate((x_merge,x_ml),axis=1)
    return x_final

  def feature_engineer(self, df):
    df['title_len'] = df.title.apply(lambda x: len(x))
    df['body_len'] = df.body.apply(lambda x: len(x))
    df['body_len'] = df['body_len'] / df['body_len'].max()
    df["domain"] = df["url"].apply(lambda x: self.extract_domain(x))
    self.lst_domain = self.get_lst_domain()
    df['caps_in_title'] = df['title'].apply(lambda title: sum(1 for char in title if char.isupper()))
    df['caps_in_title'] = df['caps_in_title'] / df['title_len']
    df['title_len'] = df['title_len'] / df['title_len'].max()
    df = df.drop('url', axis=1)
    return df

  def extract_domain(self, x):
    if x == "no_url":
      return x
    try:
        domain = x.split("://")[1].split("/")[0]
    except:
        domain = x.split("www.")[1].split("/")[0]
    return domain.split(".")[0]

  def get_lst_domain(self):
    vc = self.df["domain"].value_counts()
    lst_domain = list(vc[vc > 2].index)
    return lst_domain
