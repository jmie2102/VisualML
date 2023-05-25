from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder

from kivy.uix.popup import Popup

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import pandas as pd

import matplotlib.pyplot as plt

import os

import ML_Pipeline_Script as pipeline

df_path_dict={'df_path':''}

class Upload():

    def getDatasetName(self,fpath):
        file_name=os.path.basename(fpath)
        self.ids.dname_val.text=file_name

        return file_name
    
    def getRowColNum(self,df):
        rows=str(df.shape[0])
        cols=str(df.shape[1])
        
        self.ids.drow_val.text=rows
        self.ids.dcol_val.text=cols

        return rows,cols

    def getColNames(self,df):
        col_names=[]
        col_names_disp=[]
        
        for col in df.columns:
            col_names.append(col)

        if df.shape[1]>=4:
            for i in range(0,4):
                col_names_disp.append(col_names[i])
            col_names_disp.append('.....')
            self.ids.dcolName_val.text=str(col_names_disp)
        else:
            self.ids.dcolName_val.text=str(col_names)
        
        return col_names
       
class KnnPopup(Popup):
    def TrainKnn(self,n_neighbors=5,leaf_size=30,weights='uniform',algorithm='auto'):
        pipeline.get_model_param(int(n_neighbors),int(leaf_size),weights,algorithm)
        print(pipeline._pipeline_info_dict['param'])

class SVMPopup(Popup):
    def TrainSVM(self,degree=3,max_iter_SVM=-1,random_state=None,kernel='auto'):

        print(degree,max_iter_SVM,random_state,kernel)
        
class NBpopup(Popup):
    def TrainNB(self,priors=None,var_smoothing=1e-9):

        print(priors,var_smoothing)
        
class LogRegPopup(Popup):
    def TrainLogReg(self,max_iterint_LogReg=100,random_state_LogReg=0,penalty_spinner='l2',solver_spinner='lbfgs',multi_class_spinner='auto'):
        pipeline.get_model_param(int(max_iterint_LogReg),int(random_state_LogReg),penalty_spinner,solver_spinner,multi_class_spinner)
        print(pipeline._pipeline_info_dict['param'])

class DTPopup(Popup):
    def TrainDT(self,max_depth_DT=None,random_state=None,criterio_DT='gini',splitter='rbf'):
        
        print(max_depth_DT,random_state,criterio_DT,splitter)

class RFPopup(Popup):
    def TrainRF(self,n_estimators=100,max_depth_RF=None,random_state=None,criterio_RF='gini',):
        
        print(n_estimators,max_depth_RF,random_state,criterio_RF)
        
class EDATab(TabbedPanel):
    pass

class PanelDesign(TabbedPanel,Upload):

    def DisplayInfo(self,path):

        df=pd.read_csv(path)

        file_name=self.getDatasetName(path)
        rows,cols=self.getRowColNum(df)[0],self.getRowColNum(df)[1]
        col_names=self.getColNames(df)

        print('Name:',file_name)
        print('No. of rows and columns:'+'('+rows+','+cols+')')
        print('Cols names:',col_names)

    def SelectedFileChooser(self,filename):
        file_path=filename[0]

        pipeline.get_df_path(file_path)
        print(pipeline._pipeline_info_dict['path'])
        df_path_dict['df_path']=file_path
        self.DisplayInfo(file_path)

        return file_path

    #Choose the file from the system
    def chooseDatasetFolder(self):
        Tk().withdraw()

        #Locating the csv file
        print('Select your dataset (.csv) ...')

        file_path=askopenfilename(title = "Select file",filetypes=(("CSV Files","*.csv"),))
        if file_path=='':
            print('Please select a dataset')
        else:
            self.DisplayInfo(file_path)
        pipeline.get_df_path(file_path)
        print(pipeline._pipeline_info_dict['path'])
        self.ids.datasetPath.text='Path: '+file_path

        df_path_dict['df_path']=file_path

    #Checkbox to verify if the user wants to use built-in data
    def CheckBuiltin(self,value):
        if value==True:
            self.ids.uploadBtn.disabled=True
            self.ids.filechooser.disabled=False
        else:
            self.ids.uploadBtn.disabled=False
            self.ids.filechooser.disabled=True

    def ColToDrop(self,value):
        pipeline.get_drop_cols(value)
        print(pipeline._pipeline_info_dict['col_to_drop'])
    
    #Choose the imputation technique
    def ImpTech(self,value):
        pipeline.get_imp_tech(value)
        self.ids.imputation_label.text=f'Imputation Technique: {value}'

    def ImpCol(self):
        pipeline.get_imp_col(self.ids.col_imp_ip.text)
        print(pipeline._pipeline_info_dict['imp_cols'])

    def CatToNum(self):
        pipeline.get_cat_to_num_col(self.ids.cat_to_num_ip.text)
        print(pipeline._pipeline_info_dict['cat_to_num_col'])
    
    #Choose scaling technique
    def ScalingTech(self,value):
        pipeline.get_scaling_tech(value)
        self.ids.scaling_label.text=f'Scaling Technique: {value}'
    
    def load_data(self):
        df=pd.read_csv(df_path_dict['df_path'])
        
        data_label = self.ids.data_label
        data_label.text = (df.head(17)).to_string()

    def fea_null(self):
        df=pd.read_csv(df_path_dict['df_path'])

        f1=int(self.ids.features1.text)
        f2=int(self.ids.features2.text)
        fn=df.iloc[:,f1:f2].isna().sum().sum()
        self.ids.featureNull.text=str(fn)

    def lab_null(self):
        df=pd.read_csv(df_path_dict['df_path'])
        
        l1=int(self.ids.labels.text)
        ln=df.loc[l1].isna().sum().sum()
        self.ids.labelNull.text=str(ln)

    def lab_uni(self):
        df=pd.read_csv(df_path_dict['df_path'])

        l2=int(self.ids.labels.text)
        lu=df.iloc[:,l2].nunique()
        self.ids.labelUnique.text=str(lu)

    def viewbtn(self):
        df=pd.read_csv(df_path_dict['df_path'])

        f_1=int(self.ids.features1.text)
        f_2=int(self.ids.features2.text)
        l_1=int(self.ids.labels.text)

        pipeline.get_features_label(f_1,f_2,l_1)

        print(pipeline._pipeline_info_dict['x_start'],pipeline._pipeline_info_dict['x_end'],pipeline._pipeline_info_dict['y'])
    
    def linegraph(self):
        df=pd.read_csv(df_path_dict['df_path'])

        a=int(self.ids.lxaxis.text)
        b=int(self.ids.lyaxis.text)
        x=df.iloc[:,a]
        y=df.iloc[:,b]
        plt.plot(x,y)
        plt.show()

    def scattergraph(self):
        df=pd.read_csv(df_path_dict['df_path'])

        a=int(self.ids.sxaxis.text)
        b=int(self.ids.syaxis.text)
        x=df.iloc[:,a]
        y=df.iloc[:,b]
        plt.scatter(x,y)
        plt.show()

    def histograph(self):
        df=pd.read_csv(df_path_dict['df_path'])

        a=int(self.ids.haxis.text)
        x=df.iloc[:,a]
        plt.hist(x)
        plt.show()

    def bargraph(self):
        df=pd.read_csv(df_path_dict['df_path'])

        a=int(self.ids.bxaxis.text)
        b=int(self.ids.byaxis.text)
        x=df.iloc[:,a]
        y=df.iloc[:,b]
        plt.bar(x,y)
        plt.show()
    
    #Slider Function (Data Splitting)
    def slide_it(self,*args):
        self.ids.slidela.text=str(int(args[1]))

    #Confim Button for data Splitting
    def splitdata(self,slide=30,rs=0,shuffleflag=False):
        if shuffleflag=='True':
            shuffleflag=True
        else:
            shuffleflag=False
        test_size=float(slide)/100
        pipeline.get_splitting_param(float(test_size),int(rs),shuffleflag)
        print(pipeline._pipeline_info_dict['splitting_param'])
    
    #Passing the model name
    def ModelName(self,value):
        self.ids.trainingstatus.text='Training status: Finished'
        pipeline.get_model_name(value)
    
    def BeginTraining(self):
        pipeline.StartPipline()
    
    def UpdateScore(self):
        self.ids.accuracy.text=pipeline._eval_score_dict['accuracy']
        self.ids.precision.text=pipeline._eval_score_dict['precision']
        self.ids.recall.text=pipeline._eval_score_dict['recall']
        self.ids.f1.text=pipeline._eval_score_dict['f1']

        cm_plot_path='C:\\Codes\\Python\\VisualML\\figs\\cm_plot.png'
        
        self.ids.confusion_matrix.source=cm_plot_path
        
    def model_visualization(self):
        self.ids.modelvisimg.source='C:\\Codes\\Python\\VisualML\\figs\\knn.png'

class MainApp(App):
    def build(self):
        Builder.load_file('Layout.kv')
        return PanelDesign()

if __name__=='__main__':
    MainApp().run()
    