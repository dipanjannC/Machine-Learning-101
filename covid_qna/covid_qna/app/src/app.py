from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api

import os
import json
import numpy as np
import pandas as pd
import re
import gc
import requests
from bs4 import BeautifulSoup
import datetime
import dateutil.parser as dparser

import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch
import transformers
from transformers import *
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

import pyserini
from pyserini.search import pysearch
from IPython.core.display import display, HTML
from tqdm import tqdm
from Bio import Entrez, Medline
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import warnings

os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2/"
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Initailize tensorflow module globally if you have GPU else comment out this part. Please check the no_gpu branch. 
def embed_useT():
   module = '/sentence_wise_email/module/module_useT'
   with tf.Graph().as_default():
       sentences = tf.compat.v1.placeholder(tf.string)
       embed = hub.Module(module)
       embeddings = embed(sentences)
       session = tf.compat.v1.train.MonitoredSession()
   return lambda x: session.run(embeddings, {sentences: x})
embed_fn = embed_useT()


class BertSquad:

    USE_SUMMARY = True
    FIND_PDFS = False
    SEARCH_MEDRXIV = False
    SEARCH_PUBMED = False

    minDate = '2020/08/13'
    luceneDir = '/data/indexes/lucene-index-cord19/'

    torch_device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

    QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_MODEL.to(torch_device)
    QA_MODEL.eval()

    if USE_SUMMARY:
        SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        SUMMARY_MODEL.to(torch_device)
        SUMMARY_MODEL.eval()

    para_model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
    para_tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
    gc.collect()


    def reconstructText(self, tokens, start=0, stop=-1):
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList =[]
        for i,t in enumerate(txt_list):
            if i < nTxtL -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    newList += [t,',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        return ''.join(newList)


    def makeBERTSQuADPrediction(self, document, question):
        nWords = len(document.split())
        input_ids_all = self.QA_TOKENIZER.encode(question, document)
        tokens_all = self.QA_TOKENIZER.convert_ids_to_tokens(input_ids_all)
        overlapFac = 1.1
        if len(input_ids_all)*overlapFac > 2048:
            nSearchWords = int(np.ceil(nWords/5))
            quarter = int(np.ceil(nWords/4))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[quarter-int(nSearchWords*overlapFac/2):quarter+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*2-int(nSearchWords*overlapFac/2):quarter*2+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*3-int(nSearchWords*overlapFac/2):quarter*3+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
            
        elif len(input_ids_all)*overlapFac > 1536:
            nSearchWords = int(np.ceil(nWords/4))
            third = int(np.ceil(nWords/3))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[third-int(nSearchWords*overlapFac/2):third+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[third*2-int(nSearchWords*overlapFac/2):third*2+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
            
        elif len(input_ids_all)*overlapFac > 1024:
            nSearchWords = int(np.ceil(nWords/3))
            middle = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[middle-int(nSearchWords*overlapFac/2):middle+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        elif len(input_ids_all)*overlapFac > 512:
            nSearchWords = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        else:
            input_ids = [input_ids_all]
        absTooLong = False    
        
        answers = []
        cons = []
        for iptIds in input_ids:
            tokens = self.QA_TOKENIZER.convert_ids_to_tokens(iptIds)
            sep_index = iptIds.index(self.QA_TOKENIZER.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(iptIds) - num_seg_a
            segment_ids = [0]*num_seg_a + [1]*num_seg_b
            assert len(segment_ids) == len(iptIds)
            n_ids = len(segment_ids)

            if n_ids < 512:
                start_scores, end_scores = self.QA_MODEL(torch.tensor([iptIds]).to(self.torch_device),
                                        token_type_ids=torch.tensor([segment_ids]).to(self.torch_device))
            else:
                #this cuts off the text if its more than 512 words so it fits in model space 
                print('****** warning only considering first 512 tokens, document is '+str(nWords)+' words long.  There are '+str(n_ids)+ ' tokens')
                absTooLong = True
                start_scores, end_scores = self.QA_MODEL(torch.tensor([iptIds[:512]]).to(self.torch_device),
                                        token_type_ids=torch.tensor([segment_ids[:512]]).to(self.torch_device))
            start_scores = start_scores[:,1:-1]
            end_scores = end_scores[:,1:-1]
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            answer = self.reconstructText(tokens, answer_start, answer_end+2)
        
            if answer.startswith('. ') or answer.startswith(', '):
                answer = answer[2:]
                
            c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()
            answers.append(answer)
            cons.append(c)
        
        maxC = max(cons)
        iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]
        confidence = cons[iMaxC]
        answer = answers[iMaxC]
        
        sep_index = tokens_all.index('[SEP]')
        full_txt_tokens = tokens_all[sep_index+1:]
        
        abs_returned = self.reconstructText(full_txt_tokens)

        ans={}
        ans['answer'] = answer
        if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):
            ans['confidence'] = -1000000
        else:
            ans['confidence'] = confidence
        ans['abstract_bert'] = abs_returned
        ans['abs_too_long'] = absTooLong
        return ans


    def searchAbstracts(self, hit_dictionary, question):
        abstractResults = {}
        for k,v in tqdm(hit_dictionary.items()):
            abstract = v['abstract_full']
            if abstract:
                ans = self.makeBERTSQuADPrediction(abstract, question)
                if ans['answer']:
                    confidence = ans['confidence']
                    abstractResults[confidence]={}
                    abstractResults[confidence]['main_abstract'] = abstract
                    abstractResults[confidence]['answer'] = ans['answer']
                    abstractResults[confidence]['abstract_bert'] = ans['abstract_bert']
                    abstractResults[confidence]['idx'] = k
                    abstractResults[confidence]['abs_too_long'] = ans['abs_too_long']
                    
        cList = list(abstractResults.keys())
        if cList:
            maxScore = max(cList)
            total = 0.0
            exp_scores = []
            for c in cList:
                s = np.exp(c-maxScore)
                exp_scores.append(s)
            total = sum(exp_scores)
            for i,c in enumerate(cList):
                abstractResults[exp_scores[i]/total] = abstractResults.pop(c)
        return abstractResults


    def displayResults(self, hit_dictionary, answers, question):
        
        question_HTML = '<div font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        confidence = list(answers.keys())
        confidence.sort(reverse=True)

        for c in confidence:
            if c>0 and c <= 1 and len(answers[c]['answer']) != 0:
                if 'idx' not in  answers[c]:
                    continue
                rowData = []
                idx = answers[c]['idx']
                title = hit_dictionary[idx]['title']
                authors = hit_dictionary[idx]['authors'] + ' et al.'
                doi = '<a href="https://doi.org/'+hit_dictionary[idx]['doi']+'" target="_blank">' + title +'</a>'
                main_abstract = answers[c]['main_abstract']
                
                full_abs = answers[c]['abstract_bert']
                bert_ans = answers[c]['answer']
                
                split_abs = full_abs.split(bert_ans)
                sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
                if len(split_abs) == 1:
                    sentance_end_pos = len(full_abs)
                    sentance_end =''
                else:
                    sentance_end_pos = split_abs[1].find('. ')+1
                    if sentance_end_pos == 0:
                        sentance_end = split_abs[1]
                    else:
                        sentance_end = split_abs[1][:sentance_end_pos]
                    
                answers[c]['full_answer'] = sentance_beginning+bert_ans+sentance_end
                answers[c]['sentence_beginning'] = sentance_beginning
                answers[c]['sentence_end'] = sentance_end
                answers[c]['title'] = title
                answers[c]['doi'] = doi
                answers[c]['main_abstract'] = main_abstract
                if 'pdfLink' in hit_dictionary[idx]:
                    answers[c]['pdfLink'] = hit_dictionary[idx]['pdfLink']

            else:
                answers.pop(c)
        
        # Please check the no_gpu branch. Comment out this part if the system doesn't support GPU
        ## re-rank based on semantic similarity of the answers to the question
        cList = list(answers.keys())
        allAnswers = [answers[c]['full_answer'] for c in cList]

        messages = [question]+allAnswers

        encoding_matrix = embed_fn(messages)
        gc.collect()
        similarity_matrix = np.inner(encoding_matrix, encoding_matrix)
        rankings = similarity_matrix[1:, 0]

        for i, c in enumerate(cList):
            answers[rankings[i]] = answers.pop(c)
            
        # Comment till here if required

        ## now form pandas dv
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        pandasData = []
        ranked_aswers = []
        for c in confidence:
            rowData=[]
            title = answers[c]['title']
            main_abstract = answers[c]['main_abstract']
            doi = answers[c]['doi']
            idx = answers[c]['idx']
            rowData += [idx]            
            sentance_html = '<div>' +answers[c]['sentence_beginning'] + " <font color='#08A293'>"+answers[c]['answer']+"</font> "+answers[c]['sentence_end']+'</div>'
            
            rowData += [sentance_html, c, doi, main_abstract]
            pandasData.append(rowData)
            ranked_aswers.append(' '.join([answers[c]['full_answer']]))
        
        if self.FIND_PDFS or self.SEARCH_MEDRXIV:
            pdata2 = []
            for rowData in pandasData:
                rd = rowData
                idx = rowData[0]
                if 'pdfLink' in answers[rowData[2]]:
                    rd += ['<a href="'+answers[rowData[2]]['pdfLink']+'" target="_blank">PDF Link</a>']
                elif self.FIND_PDFS:
                    if str(idx).startswith('pm_'):
                        pmid = idx[3:]
                    else:
                        try:
                            test = self.UrlReverse('https://doi.org/'+hit_dictionary[idx]['doi'])
                            if test is not None:
                                pmid = test.pmid
                            else:
                                pmid = None
                        except:
                            pmid = None
                    pdfLink = None
                    if pmid is not None:
                        try:
                            pdfLink = self.FindIt(str(pmid))
                        except:
                            pdfLink = None
                    if pdfLink is not None:
                        pdfLink = pdfLink.url

                    if pdfLink is None:

                        rd += ['Not Available']
                    else:
                        rd += ['<a href="'+pdfLink+'" target="_blank">PDF Link</a>']
                else:
                    rd += ['Not Available']
                pdata2.append(rowData)
        else:
            pdata2 = pandasData

        df = pd.DataFrame(pdata2, columns=['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
        
        if self.USE_SUMMARY:
            allAnswersTxt = ' '.join(ranked_aswers[:6]).replace('\n','')
            answers_input_ids = self.SUMMARY_TOKENIZER.batch_encode_plus([allAnswersTxt], return_tensors='pt', max_length=1024)['input_ids'].to(self.torch_device)
            summary_ids = self.SUMMARY_MODEL.generate(answers_input_ids, num_beams=10, length_penalty=1.2, max_length=1024, min_length=64, no_repeat_ngram_size=4)

            exec_sum = self.SUMMARY_TOKENIZER.decode(summary_ids.squeeze(), skip_special_tokens=True)
            execSum_HTML = '<div style="font-size:12px;color:#CCCC00"><b>BART Abstractive Summary:</b>: '+exec_sum+'</div>'
            warning_HTML = '<div style="font-size:12px;padding-bottom:12px;color:#CCCC00;margin-top:1px"> Warning: This is an autogenerated summary based on semantic search of abstracts, please examine the results before accepting this conclusion. There may be scenarios in which the summary will not be able to clearly answer the question.</div>'
        
        if self.FIND_PDFS or self.SEARCH_MEDRXIV:
            df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
        else:
            df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
            
        return exec_sum, warning_HTML, df.to_json(orient="records", force_ascii=True, default_handler=None)


    def getrecord(self, id, db):
        handle = Entrez.efetch(db=db, id=id, rettype='Medline', retmode='text')
        rec = handle.read()
        handle.close()
        return rec


    def pubMedSearch(self, terms, db='pubmed', mindate='2019/12/01'):
        handle = Entrez.esearch(db = db, term = terms, retmax=10, mindate=mindate)
        record = Entrez.read(handle)
        record_db = {}
        for id in record['IdList']:
            try:
                record = self.getrecord(id,db)
                recfile = StringIO(record)
                rec = Medline.read(recfile)
                if 'AB' in rec and 'AU' in rec and 'LID' in rec and 'TI' in rec:
                    if '10.' in rec['LID'] and ' [doi]' in rec['LID']:
                        record_db['pm_'+id] = {}
                        record_db['pm_'+id]['authors'] = ' '.join(rec['AU'])
                        record_db['pm_'+id]['doi'] = '10.'+rec['LID'].split('10.')[1].split(' [doi]')[0]
                        record_db['pm_'+id]['abstract'] = rec['AB']
                        record_db['pm_'+id]['title'] = rec['TI']
            except:
                print("Problem trying to retrieve: " + str(id))
            
        return record_db
    Entrez.email = 'pubmedkaggle@gmail.com'

   
    def medrxivSearch(self, query, n_pages=1):
        results = {}
        q = query
        for x in range(n_pages):
            PARAMS = {
                'page': x
            }
            r = requests.get('https://www.medrxiv.org/search/' + q, params = PARAMS)
            content = r.text
            page = BeautifulSoup(content, 'lxml')
            
            for entry in page.find_all("a", attrs={"class": "highwire-cite-linked-title"}):
                title = ""
                url = ""
                pubDate = ""
                journal = None
                abstract = ""
                authors = []
                database = "medRxiv"
                
                url = "https://www.medrxiv.org" + entry.get('href')
                
                request_entry = requests.get(url)
                content_entry = request_entry.text
                page_entry = BeautifulSoup(content_entry, 'lxml')
                doi = page_entry.find("span", attrs={"class": "highwire-cite-metadata-doi"}).text.split('doi.org/')[-1]

                #getting pubDate
                pubDate = page_entry.find_all("div", attrs = {"class": "pane-content"})
                pubDate = pubDate[10]
                pubDate = str(dparser.parse(pubDate, fuzzy = True))
                pubDate = datetime.datetime.strptime(pubDate, '%Y-%m-%d %H:%M:%S')
                pubDate = pubDate.strftime('%b %d %Y')
                date = pubDate.split()
                month = date[0]
                day = date[1]
                year = date[2]
                pubDate = {
                    'year': year,
                    'month': month,
                    'day': day
                }

                #getting title
                title = page_entry.find("h1", attrs={"class": "highwire-cite-title"}).text
                #getting abstract
                abstract = page_entry.find("p", attrs = {"id": "p-2"}).text.replace('\n', ' ')
                #getting authors 
                givenNames = page_entry.find_all("span", attrs={"class": "nlm-given-names"})
                surnames = page_entry.find_all("span",  attrs={"class": "nlm-surname"})
                names = list(zip(givenNames,surnames))
                for author in names:
                    name = author[0].text + ' ' + author[1].text
                    if name not in authors:
                        authors.append(name)
                
                result = {
                    'title': title,
                    'url': url,
                    'pubDate': pubDate,
                    'journal': journal,
                    'abstract': abstract,
                    'authors': authors[0],
                    'database': database,
                    'doi': doi,
                    'pdfLink': url+'.full.pdf'
                }
                results['mrx_'+result['doi'].split('/')[-1]] = result
                #break

        return results


    def searchDatabase(self, question, keywords, pysearch):
        ## search the lucene database with a combination of the question and the keywords
        pm_kw = ''
        minDate='2019/12/01'
        k=20
        
        searcher = pysearch.SimpleSearcher(self.luceneDir)
        hits = searcher.search(question + '. ' + keywords, k=k)
        n_hits = len(hits)
        ## collect the relevant data in a hit dictionary
        hit_dictionary = {}
        for i in range(0, n_hits):
            doc_json = json.loads(hits[i].raw)
            idx = str(hits[i].docid)
            hit_dictionary[idx] = doc_json
            hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
            hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
            hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")
            
        titleList = [h['title'] for h in hit_dictionary.values()]
        
        # search for PubMed and medArxiv data dynamically
        if pm_kw:
            if SEARCH_PUBMED:
                new_hits = pubMedSearch(pm_kw, db='pubmed', mindate=minDate)
                for id,h in new_hits.items():
                    if h['title'] not in titleList:
                        titleList.append(h['title'])
                    hit_dictionary[id] = h
            if SEARCH_MEDRXIV:
                new_hits = medrxivSearch(pm_kw)
                for id,h in new_hits.items():
                    if h['title'] not in titleList:
                        titleList.append(h['title'])
                    hit_dictionary[id] = h
        
        ## scrub the abstracts in prep for BERT-SQuAD
        for idx,v in hit_dictionary.items():

            try:
                abs_dirty = v['abstract']
            except KeyError:
                print("Sorry! No abstract found.")
                abs_dirty = ''
                # uncomment the code if required search on body_text also. Will impact processing time

    #             if v['has_full_text'] == True:
    #                 print(v['paper_id'])
    #                 abs_dirty = v['body_text']
    #             else:
    #                 print(v.keys())
    #         abs_dirty = ''
    #         abs_dirty = v['abstract']

            # looks like the abstract value can be an empty list
            v['abstract_paragraphs'] = []
            v['abstract_full'] = ''

            if abs_dirty:
                # if it is a list, then the only entry is a dictionary where text is in 'text' key it is broken up by paragraph if it is in that form.  
                # make lists for every paragraph that is full abstract text as both could be valuable for BERT derrived QA

                if isinstance(abs_dirty, list):
                    for p in abs_dirty:
                        v['abstract_paragraphs'].append(p['text'])
                        v['abstract_full'] += p['text'] + ' \n\n'

                # in some cases the abstract can be straight up text so we can actually leave that alone
                if isinstance(abs_dirty, str):
                    v['abstract_paragraphs'].append(abs_dirty)
                    v['abstract_full'] += abs_dirty + ' \n\n'
        
        ## Search collected abstracts with BERT-SQuAD
        answers = self.searchAbstracts(hit_dictionary, question)
        ## displaying results in a nice format
        return self.displayResults(hit_dictionary, answers, question)


    def show_query(self, query):
        """HTML print format for the searched query"""
        return HTML('<br/><div font-size: 20px;'
                    'padding-bottom:12px"><b>Query</b>: ' + query + '</div>')

    def show_document(self, idx, doc):
        """HTML print format for document fields"""
        have_body_text = 'body_text' in json.loads(doc.raw)
        body_text = ' Full text available.' if have_body_text else ''
        return HTML('<div font-size: 18px; padding-bottom:10px">' +
                    f'<b>Document {idx}:</b> {doc.docid} ({doc.score:1.2f}) -- ' +
                    f'{doc.lucene_document.get("authors")} et al. ' +
                    f'{doc.lucene_document.get("title")}. ' +
                    f'<a href="https://doi.org/{doc.lucene_document.get("doi")}">{doc.lucene_document.get("doi")}</a>.'
                    + f'{body_text}</div>')

    def extract_scibert(self, text, tokenizer, model):
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]
        n_chunks = int(np.ceil(float(text_ids.size(1)) / 510))
        states = []
        for ci in range(n_chunks):
            text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
            text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
            if text_ids[0, -1] != text_ids[0, -1]:
                text_ids_ = torch.cat([text_ids_, text_ids[0, -1].unsqueeze(0)])
            with torch.no_grad():
                state = model(text_ids_.unsqueeze(0))[0]
                state = state[:, 1:-1, :]
            states.append(state)
        state = torch.cat(states, axis=1)
        return text_ids, text_words, state[0]

    def get_result_id(self, query, doc_id, searcher):
        """HTML print format for the searched query"""
        hits = searcher.search(query)
        display(self.show_query(query))
        for i, hit in enumerate(hits):
            if hit.docid == doc_id:
                display(self.show_document(i + 1, hit))
                return hit

    def cross_match(self, state1, state2):
        state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))
        state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))
        sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)
        return sim

    def show_sections(self, section, text):
        """HTML print format for document subsections"""
        return HTML(
            '<div font-size: 18px; padding-bottom:10px; margin-left: 15px">' +
            f'<b>{section}</b> -- {text.replace(" ##", "")} </div>')

    def highlight_paragraph(self, ptext, rel_words, max_win=10):
        para = ""
        prev_idx = 0
        for jj in rel_words:
            if prev_idx > jj:
                continue
            found_start = False
            for kk in range(jj, prev_idx - 1, -1):
                if ptext[kk] == "." and (ptext[kk + 1][0].isupper() or ptext[kk + 1][0] == '['):
                    sent_start = kk
                    found_start = True
                    break
            if not found_start:
                sent_start = prev_idx - 1
            found_end = False
            for kk in range(jj, len(ptext) - 1):
                if ptext[kk] == "." and (ptext[kk + 1][0].isupper() or ptext[kk + 1][0] == '['):
                    sent_end = kk
                    found_end = True
                    break
            if not found_end:
                if kk >= len(ptext) - 2:
                    sent_end = len(ptext)
                else:
                    sent_end = jj
            para = para + " "
            para = para + " ".join(ptext[prev_idx:sent_start + 1])
            para = para + " <font color='blue'>"
            para = para + " ".join(ptext[sent_start + 1:sent_end])
            para = para + "</font> "
            prev_idx = sent_end
        if prev_idx < len(ptext):
            para = para + " ".join(ptext[prev_idx:])
        return para

    def show_results(self, question, doc_id):
        searcher = pysearch.SimpleSearcher(self.luceneDir)
        query = (question)
        highlighted_text = ""
        query_ids, query_words, query_state = self.extract_scibert(query, self.para_tokenizer, self.para_model)
        req_doc = json.loads(self.get_result_id(query, doc_id, searcher).raw)
        paragraph_states = []
        for par in tqdm(req_doc['body_text']):
            state = self.extract_scibert(par['text'], self.para_tokenizer, self.para_model)
            paragraph_states.append(state)
        sim_matrices = []
        for pid, par in tqdm(enumerate(req_doc['body_text'])):
            sim_score = self.cross_match(query_state, paragraph_states[pid][-1])
            sim_matrices.append(sim_score)
        paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]

        # Select the index of top 5 paragraphs with highest relevance
        rel_index = np.argsort(paragraph_relevance)[-5:][::-1]
        for ri in np.sort(rel_index):
            sim = sim_matrices[ri].data.numpy()

            # Select the two highest scoring words in the paragraph
            rel_words = np.sort(np.argsort(sim.max(0))[-2:][::-1])
            p_tokens = paragraph_states[ri][1]
            para = self.highlight_paragraph(p_tokens, rel_words)
            highlighted_text += para
            display(self.show_sections(req_doc["body_text"][ri]['section'], para))
        data = {'id': doc_id, 'title': req_doc['metadata']['title'], 'text': highlighted_text}
        return data


class Initialize(Resource):
    def get(self):
        message = {'message': 'Hello World!'}
        return message, 200

    def post(self):
        json_data = request.get_json()
        message = {'message': json_data}
        return message, 201


class GetAnswerBert(Resource):
    def post(self):
        json_data = request.get_json()
        api = BertSquad()
        kw_list = ""
        question = json_data['question']
        
        rich_text, warn, result = api.searchDatabase(question, kw_list, pysearch)
        message = {'rich_text': rich_text, 'warning': warn, 'result': result}
        return message, 200


class GetDetailAnswerBert(Resource):
    def post(self):
        json_data = request.get_json()
        api = BertSquad()
        question = json_data['question']
        doc_id = json_data['doc_id']
        result = api.show_results(question, doc_id)
        return result, 200


api.add_resource(Initialize, '/')
api.add_resource(GetAnswerBert, '/rich-text')
api.add_resource(GetDetailAnswerBert, '/detailed-text')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)

# Use the below command for testing
# curl -H "Content-Type: application/json" -X POST -d '{"question": "Hi"}' http://127.0.0.1:5000/
