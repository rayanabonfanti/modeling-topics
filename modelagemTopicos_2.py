#Gerencias as matrizes, leitura e escrita em arquivos do Excel
import pandas as pd
#Realiza cálculos numéricos
import numpy as np
#Calcula matriz de termos dos documentos
import sklearn.feature_extraction as fe
#Implementa classe para LDA
import sklearn.decomposition as sd
#Avalia expressões regulares
import regex as re
#Classes e funções para processamento de linguagem natural
import nltk as nt

artigos = pd.read_csv("dados/scopus_exemplo_1.csv")

nt.download('punkt')
nt.download('stopwords')

stemmer = nt.stem.snowball.EnglishStemmer(ignore_stopwords=True)

#Especialização da classe que gera DTM para levar em consideração flexões de palavras
class StemmedCountVectorizer(fe.text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def preprocessador(texto):
  #transforma palavras para minúsculo
  texto = texto.lower()
  #remove caracteres especiais #$%@ etc
  texto = re.sub("\\W"," ",texto)
  #Remove números
  texto = re.sub("(\s[0-9]+|[0-9]\s)"," ",texto)
  return texto.split()

contador = StemmedCountVectorizer(stop_words='english',tokenizer=preprocessador)
dtm = contador.fit_transform(artigos["Abstract"])

dtm.shape

lda = sd.LatentDirichletAllocation(n_components=5)
analise_topicos = lda.fit(dtm)
nomes = contador.get_feature_names()
componentes = pd.DataFrame(analise_topicos.components_,columns=nomes)
componentes

n_palavras = 10
topicos = []
for i in range(componentes.shape[0]):
  topicos.append(componentes.iloc[i].sort_values(ascending=False)[0:n_palavras-1])

grafico_1 = topicos[1].plot.barh(title="Tópico 1")
grafico_1.set_xlabel("")
grafico_1.set_ylabel("Palavras")
grafico_1.plot()

grafico_2 = topicos[2].plot.barh(title="Tópico 2")
grafico_2.set_xlabel("")
grafico_2.set_ylabel("Palavras")
grafico_2.plot()

grafico_3 = topicos[3].plot.barh(title="Tópico 3")
grafico_3.set_xlabel("")
grafico_3.set_ylabel("Palavras")
grafico_3.plot()
