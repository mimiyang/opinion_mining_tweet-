import xlrd
import sys
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from sklearn import svm, metrics
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score
#import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

excel_sheets=[]
d_excel={}
d_data={}

excel_sheets1=[]
d_excel1={}
d_data1={}

F1_score={}
Confusion_matrix={}

def import_file_train(data_file):
	workbook = xlrd.open_workbook(data_file)
        for sheet in workbook.sheets():
                con_sheet=[]
                c_sheet=[]
		d_sheet=[]
                for r in range(sheet.nrows):
                    if r >=2:
			v_row=[]
			for c in range(sheet.ncols):
                           if c==3 or c == 4:
				t_cell=sheet.cell_type(r, c)
				if t_cell == xlrd.XL_CELL_NUMBER:
					v_row.append(str(0))
				if t_cell == xlrd.XL_CELL_NUMBER:
					v_row.append(int(sheet.cell_value(r, c)))
				else:
					v_row.append(sheet.cell_value(r,c))
			if len(v_row)>=3 and v_row[2] in [1,0,-1]:
                                
                                v_row=[v_row[0],v_row[2]]
                                d_sheet.append(v_row)
                	
		if len(d_sheet)>0:
                        for row in d_sheet:
                                lower = unicode(row[0]).encode('ascii','ignore')
                                lower = lower.lower()
                                lower.translate(None, string.punctuation)
                                con_sheet.append(lower)
                                c_sheet.append(row[1])
                d_excel1[sheet.name]=[con_sheet,c_sheet]
                excel_sheets1.append(sheet.name)
                

        #print('read excel successfully')

def import_file_test(data_file):
	workbook = xlrd.open_workbook(data_file)
        for sheet in workbook.sheets():
                con_sheet=[]
                c_sheet=[]
		d_sheet=[]
                for r in range(sheet.nrows):
                    if r >=2:
			v_row=[]
			for c in range(sheet.ncols):
                           if c==0 or c == 4:
				t_cell=sheet.cell_type(r, c)
				if t_cell == xlrd.XL_CELL_NUMBER:
					v_row.append(str(0))
				if t_cell == xlrd.XL_CELL_NUMBER:
					v_row.append(int(sheet.cell_value(r, c)))
				else:
					v_row.append(sheet.cell_value(r,c))
			if len(v_row)>=3 and v_row[2] in [1,0,-1]:
                                
                                v_row=[v_row[0],v_row[2]]
                                d_sheet.append(v_row)
                	
		if len(d_sheet)>0:
                        for row in d_sheet:
                                lower = unicode(row[0]).encode('ascii','ignore')
                                lower = lower.lower()
                                lower.translate(None, string.punctuation)
                                con_sheet.append(lower)
                                c_sheet.append(row[1])
                d_excel[sheet.name]=[con_sheet,c_sheet]
                excel_sheets.append(sheet.name)
                

        #print('read excel successfully')
                
def Tfidf_feature_test():
        refine_swords=[]
        for v in stopwords.words('english'):
                if (v!=u'not' and v!=u'no' and v!='don' and v!='nor'):
                    refine_swords.append(v)
        #print refine_swords
        for i in excel_sheets:
                con_sheet=d_excel[i]
                if len(con_sheet[0])>0:
                        tfidf_value=TfidfVectorizer(min_df=2,ngram_range=(1,1),stop_words=refine_swords)
                        X=tfidf_value.fit_transform(con_sheet[0]).toarray()
                        Y=con_sheet[1]
                        d_data[i]=[X,Y]
                       # print tfidf_value.get_feature_names()
                con_sheet1=d_excel1[i]
        
        
def Tfidf_feature_train():
        tfidf_value={}
        for i in excel_sheets1:
                
                con_sheet1=d_excel1[i]
                
                con_sheet = d_excel[i]
                m_test=len(con_sheet[0])
                m_train=len(con_sheet1[0])
                tem_X = con_sheet[0]
                tem_X1 = con_sheet1[0]
                tem_X.extend(tem_X1)
                
                
                print i
                print m_test
                print m_train
                if len(con_sheet[0])>0:
                        tfidf_value[i]=TfidfVectorizer(min_df=2,ngram_range=(1,1))
                        X_to=tfidf_value[i].fit_transform(tem_X).toarray()
                        
                        X1=X_to[m_test:]
                        Y1=con_sheet1[1]
                        print len(Y1)
                        X=X_to[0:m_test]
                        Y=con_sheet[1]
                        d_data[i]=[X,Y]
                       # print tfidf_value.get_feature_names()
                        d_data1[i]=[X1,Y1]
                        print len(X[1])
                        print len(Y)
                        print len(d_data1[i][0])
                       
        
        

def classifier_class():
        print d_data.keys()
        print d_data1.keys()
        key00 = d_data.keys()[0]
        print d_excel[key00][0][0]
        print d_excel[key00][1][0]

        
        
        for key in d_data.keys():
                print key
                X=d_data[key][0]
                Y=d_data[key][1]
                X1=d_data1[key][0]
                Y1=d_data1[key][1]
                
                print len(X[0])
                print Y1[0:5]
                print len(X1)
                print len(X)
                m = len(X)/3
                print d_excel1[key][0][0]
                print d_excel1[key][1][0]
                print('\n\nThe classification information for '+ key + ' Dataset is shown below')
                clf00 = svm.SVC(kernel='rbf',gamma=0.58, C=0.81)
                
                clf01=clf00.fit(X1,Y1)
                predicted_Y=clf01.predict(X)
                F1_score[key]=f1_score(Y, predicted_Y, average=None)
                Confusion_matrix[key]=confusion_matrix(Y, predicted_Y)
                
                print(Confusion_matrix[key])
                print metrics.classification_report(Y,predicted_Y)
      
  
                
def main():
	import_file_test(sys.argv[1])
	import_file_train(sys.argv[2])
        #Tfidf_feature_test()
        Tfidf_feature_train()
        classifier_class()

if __name__ == "__main__":main()
