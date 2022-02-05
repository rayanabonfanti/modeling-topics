from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
#para stopwords em Português
#import nltk
import numpy as np
import sklearn.feature_extraction as fe
import sklearn.decomposition as sd
import regex as re
import nltk as nt
import sklearn.cluster as sc
import wordcloud as wc

artigos = pd.read_csv("dados/scopus.csv")

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

grafico_1 = topicos[1].plot.barh(title="Tópico 1 - Computação")
grafico_1.set_xlabel("")
grafico_1.set_ylabel("Palavras")
grafico_1.plot()

grafico_2 = topicos[2].plot.barh(title="Tópico 2 - Educacional")
grafico_2.set_xlabel("")
grafico_2.set_ylabel("Palavras")
grafico_2.plot()

grafico_3 = topicos[3].plot.barh(title="Tópico 3 - Aprendizado/Design")
grafico_3.set_xlabel("")
grafico_3.set_ylabel("Palavras")
grafico_3.plot()

grafico_4 = topicos[4].plot.barh(title="Tópico 4 - Educacional")
grafico_4.set_xlabel("")
grafico_4.set_ylabel("Palavras")
grafico_4.plot()

grafico_5 = topicos[0].plot.barh(title="Tópico 5 - Educacional")
grafico_5.set_xlabel("")
grafico_5.set_ylabel("Palavras")
grafico_5.plot()

kmeans = sc.KMeans(n_clusters=analise_topicos.n_components)
topicos_documentos = analise_topicos.transform(dtm)
agrupamento = kmeans.fit(topicos_documentos)
#agrupamento.labels_

agrupamento.inertia_

kmeans2 = sc.KMeans(n_clusters=analise_topicos.n_components)
#topicos_documentos = analise_topicos.transform(dtm)
agrupamento2 = kmeans.fit(dtm)
#agrupamento2.labels_

agrupamento2.inertia_

grupo_1 = artigos[agrupamento.labels_ == 0]
grupo_1["Title"][0:5]

abs = grupo_1["Title"]
resumos = ''.join(abs).upper()
wordcloud = wc.WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(wc.STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

grupo_2 = artigos[agrupamento.labels_ == 1]
abs = grupo_2["Abstract"]
resumos = ''.join(abs).upper()
wordcloud = wc.WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(wc.STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

grupo_2["Title"][0:5]

grupo_3 = artigos[agrupamento.labels_ == 2]
abs = grupo_3["Title"]
resumos = ''.join(abs).upper()
wordcloud = wc.WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(wc.STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

grupo_4 = artigos[agrupamento.labels_ == 3]
abs = grupo_4["Title"]
resumos = ''.join(abs).upper()
wordcloud = wc.WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(wc.STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

grupo_5 = artigos[agrupamento.labels_ == 4]
abs = grupo_5["Title"]
resumos = ''.join(abs).upper()
wordcloud = wc.WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(wc.STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

abs = artigos["Abstract"]
resumos = ''.join(abs).upper()
#wordcloud = WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(stopwords), min_font_size = 10).generate(resumos)
wordcloud = WordCloud(width = 800, height = 800, background_color ='white',stopwords =set(STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

termos = pd.DataFrame.from_dict(wordcloud.words_,orient='index')
termos[0:10]

fontes = artigos.groupby("Abbreviated Source Title").count()
fontes_ord = fontes.sort_values("Title", ascending=False)
fontes_ord2 = pd.DataFrame({"Soma":fontes_ord["Title"]})
fontes_ord2[0:10]

ano = artigos.groupby("Year").count()
ano_ord = ano.sort_values("Title", ascending=False)
ano_ord2 = pd.DataFrame({"Soma":ano_ord["Title"]})
ano_ord2[0:10]

autores_nomes = ",".join(artigos["Authors"])
autores_nomes = autores_nomes.replace("[No author name available]","NA")
autores = autores_nomes.split(",")
autores = pd.DataFrame({"Autores": autores,"Soma":([0]*len(autores))})
autores_soma = autores.groupby("Autores").count()
autores_ord = autores_soma.sort_values("Soma",ascending=False)
autores_ord[0:10]

citacoes = artigos.sort_values("Cited by", ascending=False)
mais_citados = citacoes[["Title","Cited by","Year","Source title","Author Keywords"]]
mais_citados[0:10]

arquivo = pd.ExcelWriter("dados/tabelas.xlsx")
mais_citados.to_excel(arquivo,'Artigos mais citados',)
autores_ord.to_excel(arquivo,'Trabalhos frequentes por autores')
fontes_ord2.to_excel(arquivo,'Trabalhos frequentes por fonte')
termos.to_excel(arquivo,'Termos mais frequentes')
arquivo.close()
