from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
#para stopwords em Português
#import nltk

#nltk.download('stopwords')
#stopwords = nltk.corpus.stopwords.words('portuguese')
#set(stopwords[0:10])

artigos = pd.read_excel("dados/webofscience_exemplo_1.xls")

abs = artigos["Article Title"]
resumos = ''.join(abs).upper()
#wordcloud = WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(stopwords), min_font_size = 10).generate(resumos)
wordcloud = WordCloud(width = 800, height = 800, background_color ='white',stopwords =set(STOPWORDS), min_font_size = 10).generate(resumos)
plt.figure(figsize = (8, 8), facecolor = None)
plt.axis("off")
plt.imshow(wordcloud)
plt.show()

termos = pd.DataFrame.from_dict(wordcloud.words_,orient='index')
termos[0:6]

fontes = artigos.groupby("Source Title").count()
fontes_ord = fontes.sort_values("Article Title", ascending=False)
fontes_ord2 = pd.DataFrame({"Soma":fontes_ord["Article Title"]})
fontes_ord2

autores_nomes = ",".join(artigos["Author Full Names"])
autores_nomes = autores_nomes.replace("[No author name available]","NA")

autores = autores_nomes.split(",")
autores = pd.DataFrame({"Autores": autores,"Soma":([0]*len(autores))})

autores_soma = autores.groupby("Autores").count()
autores_ord = autores_soma.sort_values("Soma",ascending=False)

autores_ord[0:10]

citacoes = artigos.sort_values("Cited Reference Count", ascending=False)

mais_citados = citacoes[["Article Title","Cited Reference Count","Publication Year","Source Title","Author Keywords"]]

arquivo = pd.ExcelWriter("dados/tabelas.xlsx")
mais_citados.to_excel(arquivo,'Artigos e citações')
autores_ord.to_excel(arquivo,'Trabalhos por autores')
fontes_ord2.to_excel(arquivo,'Trabalhos por fonte')
termos.to_excel(arquivo,'Termos mais frequentes')
arquivo.close()