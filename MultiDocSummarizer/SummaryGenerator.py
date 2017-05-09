import nltk
import docx
import re
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import wordnet
from nltk.wsd import lesk
import math
import numpy
from sklearn.preprocessing import normalize

class SummaryGenerator:

    Stemmer = SnowballStemmer('english')
    stopwords = nltk.corpus.stopwords.words('english')
    documentClusters = []

    documents_filtered=[]
    documents_sentences_original=[]
    documents_sentences_filtered=[]
    documents_sentences_words_filtered=[]
    documents_sentences_words=[]
    query = []
    document_centrality_feature = []
    document_term_feature=[]
    document_title_similarity_feature=[]
    document_sentence_score=[]
    documents_num_of_sent=[]


    total_num_of_sent = 0
    lines_in_summary = 0
    number_sent_cluster = 0
    number_of_documents =0
    def __init__(self):
        print(" __init__")
        self.Stemmer = SnowballStemmer('english')
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.documentClusters = []

        self.documents_filtered = []
        self.documents_sentences_original = []
        self.documents_sentences_filtered = []
        self.documents_sentences_words_filtered = []
        self.documents_sentences_words = []
        self.query = []
        self.document_centrality_feature = []
        self.document_term_feature = []
        self.document_title_similarity_feature = []
        self.document_sentence_score = []
        # title = 'Big Data Analytics'

        self.title = 'Title'

        self.documents_num_of_sent = []
        self.total_num_of_sent = 0
        self.lines_in_summary = 0
        self.number_sent_cluster = 0
        self.number_of_documents = 0

    def tokenizeAndStem(self,text):
        regexpTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # r = raw-string , \w = [a-zA-Z0-9_]
        tokens = [self.Stemmer.stem(word) for word in regexpTokenizer.tokenize(text)] #Tokenize and Stem
        #  as sentence delimiters were changed to '.' hence all we need to do is check if word if '.' or not
        return tokens



    def centralityFeatureCalculation(self,count_array):
        union_count_array = count_array.sum(0) # sum of each column
        union = union_count_array.size
        centrality = []
        for count in count_array:
            intersection = 0
            for i,elem in enumerate(count):
                if(elem !=0 and union_count_array[i]>1):
                    intersection+=1
            cf = intersection/union
            centrality.append(cf)
        return centrality


    def queryStrengthen(self,doc_count_array,features):
        number_of_terms = len(features)
        number_of_documents = len(doc_count_array)
        score_array=[]
        for i in range(0,number_of_terms):
            count = doc_count_array[:,i]
            sum_count = sum(count)
            num_doc_containing = 0
            for val in count:
                if val!=0:
                    num_doc_containing+=1
            idf = math.log(number_of_documents/num_doc_containing)+1
            sum_count/=idf
            score_array.append(sum_count)


        max_frequency = (number_of_terms*4)//100 # include top 5% terms in the query
        positions = numpy.argpartition(score_array,-max_frequency)[-max_frequency:]

        regexpTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # r = raw-string , \w = [a-zA-Z0-9_]
        filtered_title_words = [self.Stemmer.stem(word) for word in regexpTokenizer.tokenize(self.title) if word not in self.stopwords]

        print("filtered Title : ", filtered_title_words)
        filtered_title_words.extend(features[positions])
        print("filtered Title : ",filtered_title_words)
        filtered_title_set=set(filtered_title_words)
        title_filtered = " ".join(filtered_title_set) + "."
        print("Strong Query : ",title_filtered)
        return filtered_title_set

    def termFeatureCalculation(self):
        vectorizer = CountVectorizer(tokenizer=self.tokenizeAndStem)
        X = vectorizer.fit_transform(self.documents_filtered)

        doc_count_array= X.toarray()
        count_array = doc_count_array.sum(0)
        totalCount = sum(count_array)
        filtered_title_set = self.queryStrengthen(doc_count_array,numpy.asarray(vectorizer.get_feature_names()))
        total_numberOfDocuments = len(self.documents_filtered)

        for i,doc in enumerate(self.documents_sentences_words_filtered):
            term_feature=[]
            num_sent_in_doc = len(doc)
            tsf_idf_score=[]
            title_similarity_feature=[]
            for sent in doc:
                tsf_idf=0
                for word in sent:

                    # for word in synset(word) : ............................................................. use lesk word sense disambiguation
                    # synSetGeneration(sent, word, pos ) # calculate pos using pos_tag
                    position = vectorizer.vocabulary_.get(word)

                    tf = count_array[position]/(totalCount-count_array[position])
                    tsf = tf # Calculate TSF

                    word_doc_count = doc_count_array[:,position]
                    num_doc_having_word = 0
                    for word_count in word_doc_count:
                        if(word_count>=1):
                            num_doc_having_word+=1

                    idf = math.log(total_numberOfDocuments/num_doc_having_word)
                    tsf_idf += tsf*idf
                tsf_idf_score.append(tsf_idf)


                #query Similarity Calculation-------------------------------------------------------------------------------
                intersection = len(filtered_title_set.intersection(sent))
                union = len(filtered_title_set.union(sent))
                title_f = intersection / union

                title_similarity_feature.append(title_f)


            self.document_term_feature.append(normalize(numpy.asarray(tsf_idf_score)).ravel())
            self.document_title_similarity_feature.append(normalize(numpy.asarray(title_similarity_feature)).ravel())


    def synSetGeneration(self,sentence,word,pos):#NOT USED
        #tagged_sent = pos_tag(sentence.split())
        synset = lesk(sentence,word,pos=pos)

        print("Synset : ",synset,"\n",synset.lemma_names())
        #print(lesk(sent,'bank'))
        #for synset in wordnet.synsets('went'):
            #print(synset.lemma_names())

    def readFromDocFile(self,path):
        regexpTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # r = raw-string , \w = [a-zA-Z0-9_]
        doc = docx.Document(path)
        sentences_filtered=[]
        sentences_original=[]
        sentences_words=[]
        document_filtered_txt=''
        #global number_of_documents
        self.number_of_documents +=1

        num_sent_in_doc=0
        document_length=0
        len_feature = []
        position_feature = []
        proper_noun_feature=[]
        numerical_data_feature=[]
        para_position_feature=[]
        index_para=0 # this is used instead of len(docs.paragraph) bcoz some paragraphs may be empty
        for para in doc.paragraphs:
            txt= para.text
            if(txt.strip()==''):
                continue

            sentences = nltk.sent_tokenize(txt)
            num_sent_in_para = len(sentences)
            for i,sentence in enumerate(sentences):

                filtered__normal_words = [word for word in regexpTokenizer.tokenize(sentence) if word.lower() not in self.stopwords]
                filtered_words = [word.lower() for word in filtered__normal_words]
                stemmed_words = [self.Stemmer.stem(word) for word in filtered_words]
                if(len(filtered_words)==0):
                    continue
                #call sentence scoring functions-------------------------------------------------------------------------------------
                sentence_length = len(filtered_words)
                document_length+=sentence_length
                num_sent_in_doc+=1

                #Position Feature
                pf = 1-i/num_sent_in_para
                position_feature.append(pf)

                #Length Feature
                lf = sentence_length # calculate
                len_feature.append(lf)

                #Proper Noun Feature
                #Tokenize by eliminating special symbols
                tagged_sent = pos_tag(sentence.split())
                #print("Tagged Sent : ",tagged_sent)

                num_properNoun = len([word for word,pos in tagged_sent if pos == 'NNP' and re.match(r'\w+',word)]) # see if to be replaced by re.search
                pnf = num_properNoun/sentence_length
                proper_noun_feature.append(pnf)

                #Numerical Data Feature
                num_numericalData=len([word for word in filtered_words if re.search(r"\d+",word)])
                ndf = num_numericalData/sentence_length
                numerical_data_feature.append(ndf)

                #Paragraph Position Feature
                para_f = index_para
                para_position_feature.append(para_f)

                #-------------------------------------------------------------------------------------------------------------------
                filtered_sentence = " ".join(filtered_words)+"."
                document_filtered_txt=document_filtered_txt+" "+filtered_sentence
                sentences_filtered.append(filtered_sentence)
                sentences_original.append(sentence)
                sentences_words.append(stemmed_words)#sentences_words.append(filtered_words)
            index_para += 1

        vectorizer = CountVectorizer(tokenizer=self.tokenizeAndStem)
        X = vectorizer.fit_transform(sentences_filtered)
        centrality_feature = self.centralityFeatureCalculation(X.toarray())


        position_feature = normalize(numpy.asarray(position_feature))
        proper_noun_feature = normalize(numpy.asarray(proper_noun_feature))
        numerical_data_feature = normalize(numpy.asarray(numerical_data_feature))
        centrality_feature = normalize(numpy.asarray(centrality_feature))

        para_position_feature = numpy.asarray(para_position_feature)
        len_feature = numpy.asarray(len_feature)


        len_feature = normalize(((len_feature*num_sent_in_doc)/document_length))
        para_position_feature = normalize((1-(para_position_feature/index_para)))

        #Normalization and assign weights to features:
        sentence_score_array= 0.1*position_feature+\
                              0.15*proper_noun_feature+\
                              0.1*numerical_data_feature+\
                              0.125*centrality_feature+\
                              0.1*len_feature+\
                              0.1*para_position_feature

       # print("Sentence SCore Array  : \n",sentence_score_array.ravel())
        sentence_score_array = sentence_score_array.ravel()
        self.document_sentence_score.append(sentence_score_array) # ravel is used to flatten the array ans normalized returns a 2d array

        for i,value in enumerate(sentence_score_array):
            lf = len_feature.ravel()[i]
            pf = position_feature.ravel()[i]
            pnf = proper_noun_feature.ravel()[i]
            ndf = numerical_data_feature.ravel()[i]
            cf = centrality_feature.ravel()[i]
            para_f = para_position_feature.ravel()[i]
            print("Sentence : ",sentences_original[i])
            print("lf : %8f "%lf," pf : %8f"%pf," pnf : %8f"%pnf," ndf : %8f"%ndf," cf :%8f "%cf," para_f %8f"%para_f)

        self.documents_num_of_sent.append(num_sent_in_doc)
        #global total_num_of_sent
        self.total_num_of_sent+=num_sent_in_doc

        self.documents_filtered.append(document_filtered_txt)
        self.documents_sentences_original.append(sentences_original)
        self.documents_sentences_filtered.append(sentences_filtered)
        self.documents_sentences_words_filtered.append(sentences_words)



    def tfIdfCalculation(self,corpus):
        vectorizer = TfidfVectorizer(tokenizer=self.tokenizeAndStem,use_idf=True,max_features=1000,max_df=0.8)
        x=vectorizer.fit_transform(corpus)
        return x


    def KMeansClustering(self,tfidfMatrix,numberOfClusters):
        kmeans = KMeans(n_clusters=numberOfClusters,random_state=0)  # random state is fixed so that each time we get same output
                                                                     #(i.e kmeans start clustering from same element)
        kmeans.fit(tfidfMatrix)
        cluster_list = kmeans.labels_.tolist()
        clusters = self.createListOfLists(numberOfClusters)
        count = 0
        for i in cluster_list: # use enumerate instead
            clusters[i].append(count)
            count=count+1
        return clusters


    def createListOfLists(self,size):
        listOfLists = list()
        for i in range(0,size):
            listOfLists.append(list())
        return listOfLists


    def corpusClustering(self,corpus,number_of_clusters):
        tfidf_matrix = self.tfIdfCalculation(corpus)
        clusters = self.KMeansClustering(tfidf_matrix,number_of_clusters)
        return clusters



    def sentenceClustering(self,cluster):
        sentences = []
        for doc in cluster:
            sentences.extend(self.documents_sentences_filtered[doc])
        print("Cluster :",cluster,"\nSentences \n",sentences)
        num_of_sent = len(sentences)
        num_of_clusters= (num_of_sent*self.number_sent_cluster)//self.total_num_of_sent # intiially it was lines_in_summary
        tfidf_matrix = self.tfIdfCalculation(sentences)
        sentence_clusters = self.KMeansClustering(tfidf_matrix,num_of_clusters)
        print("Sentence Clusters :",sentence_clusters)
        return sentence_clusters


    def summaryGenerator(self):
        summary = []
        for cluster in self.documentClusters:
            sentence_clusters = self.sentenceClustering(cluster)
            cluster_score=0
            for sent_cluster in sentence_clusters:
                max_score = 0
                max_score_sent = ''
                maxpos=0
                maxdoc=0
                cluster_score=0

                print("\n\nSent Cluster : ",sent_cluster)
                for sent in sent_cluster:
                    cumulative_sum = 0
                    for i,doc in enumerate(cluster): # find which document does the sentence belong and its position in document
                        if(cumulative_sum+self.documents_num_of_sent[doc]>sent):
                            break
                        cumulative_sum += self.documents_num_of_sent[doc]
                    position = sent-cumulative_sum

                    score = self.document_sentence_score[doc][position]
                    print("Document ", doc, "Sent: ", sent, "\nSentence : ",
                          self.documents_sentences_original[doc][position],"\nScore : ",score)  # ,documents_sentences_filtered[doc][position],"  ")

                    cluster_score+=score
                    if(max_score<score):
                        max_score=score
                        max_score_sent = self.documents_sentences_original[doc][position]
                        maxdoc = doc
                        maxpos=position
                cluster_score/=len(sent_cluster)
                summary.append([max_score_sent,self.documents_sentences_filtered[maxdoc][maxpos],cluster_score])

            summary[-1][0]+='\n\n' # separate sentences of different document clusters
            print("Summz : ",summary)
        return summary


    def setDocPaths(self,paths):
        for path in paths:
            self.readFromDocFile(path)

    def summarize(self):
        self.termFeatureCalculation()


        for i in range(0,self.number_of_documents):
            self.document_sentence_score[i]=self.document_sentence_score[i]+0.2*self.document_term_feature[i]+0.3*self.document_title_similarity_feature[i]

        print("Input : \n",self.documents_filtered)
        print(self.documents_sentences_words_filtered)
        print(self.documents_sentences_original)
        print(self.documents_sentences_filtered)


        print("\n\n")
        for i,doc in enumerate(self.documents_sentences_original):
            print("Document :- ",i,"\n")
            for j,sent in enumerate(doc):
                print("Sentence : ",sent,"Score : ",self.document_sentence_score[i][j])
        print("\n\n")


        number_of_document_clusters = math.ceil(self.number_of_documents/2)
        print("Number Of Document Clusters : ",number_of_document_clusters)
        self.documentClusters = self.corpusClustering(self.documents_filtered,number_of_document_clusters) # Number Of Document Clusters

        # Removing 30% lowest ranking clusters
        extra_clusters = (self.lines_in_summary*30)//100

        self.number_sent_cluster = self.lines_in_summary+extra_clusters

        print("Document Number of sentences : ",self.documents_num_of_sent)
        print("Document Clusters :",self.documentClusters)
        print("Scores : \n",self.document_sentence_score,"\n")
        print("Scores Doc : \n",self.document_sentence_score[0],"\n")
        print("Scores Doc Sent: \n",self.document_sentence_score[0][1],"\n")

        summary = []
        summary = self.summaryGenerator()

        summ = numpy.asarray(summary)

        print("Summary : \n",summ)
        ordered_summary = summ[numpy.argsort(summ[:,2])]
        rev = ordered_summary[::-1]
        print("\nOrdered Summary :\n",rev)

        select_clusters = self.lines_in_summary#2*(number_sent_cluster - extra_clusters)

        print("Sent Clusters : ",self.number_sent_cluster,"SElected : ",select_clusters,"type : ",type(rev))
        fin_summary = " ".join(rev[0:select_clusters,0])
        print("Final Summary : ",fin_summary)
        return fin_summary
