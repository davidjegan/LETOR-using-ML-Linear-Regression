from sklearn.cluster import KMeans #for finding centroids with kmeans clustering 
import numpy as np #for matrix manipulation operations
import csv #for reading csv files
import math #for performing sq.root in RMS and pow operation

#initializing lambda value for normalization
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False

#selecting target values from 1st column of csv    
def GetTargetVector(filePath):
        t = []
        with open(filePath, newline=None) as f:
            reader = csv.reader(f)
            for row in reader:  
                t.append(int(row[0])) 
        return t
    
#importing target values from csv into variable rawtarget
RawTarget = GetTargetVector('Querylevelnorm_t.csv') 
    
def GenerateRawData(filePath):    
        dataMatrix = [] 
        with open(filePath, newline=None) as fi:
            reader = csv.reader(fi)
            for row in reader:
                dataRow = []
                for column in row:
                    #add features corresponding to each row of sample data 
                    dataRow.append(float(column)) 
                #push each row to data matrix to be used for learning algorithm    
                dataMatrix.append(dataRow)   
        
        #delete columns where features are equal to 0 for all rows of data
        #axis value 1 deletes the respective columns 
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9],axis=1)
        dataMatrix = np.transpose(dataMatrix)  
        return dataMatrix
        
#Importing data set from csv file 
RawData   = GenerateRawData('Querylevelnorm_X.csv')
    
#function to generate training set of target values
def GenerateTrainingTarget(rawTraining,TrainingPercent):
        TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
        t           = rawTraining[:TrainingLen]
        return t
    
#function to generate training set of input values
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
        T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
        d2 = rawData[:,0:T_len]
        return d2
    
#importing target values for training set by calling function
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
#importing features and samples for training set by calling function
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    
#checking shape of target value vector for training set
print(TrainingTarget.shape)
#checking shape of input value matrix for training set 
print(TrainingData.shape)
    
#function to generate validation set of input values
def GenerateValData(rawData, ValPercent, TrainingCount): 
        valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
        V_End = TrainingCount + valSize
        #importing validation data from the sample set after using 80 % data as training set
        dataMatrix = rawData[:,TrainingCount+1:V_End]
        return dataMatrix
    
#function to generate validation set of target values
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
        valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
        #importing validation target data from the sample set after using 80 % data as training set
        V_End = TrainingCount + valSize
        t =rawData[TrainingCount+1:V_End]
        return t
    
#importing input values for validation set by calling generate function
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
#importing target values for validation set by calling generate function
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
#checking shape of input value matrix for validation set 
print(ValDataAct.shape)
#checking shape of target value matrix for validation set 
print(ValData.shape)
    
#importing input values for test set by calling generate function
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
#importing target values for test set by calling generate function
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
#checking shape of input value matrix for test set 
print(ValDataAct.shape)
#checking shape of target value matrix for test set 
print(ValData.shape)
    
#function for calculating variance of input samples with themselves
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
       #defining the size of design matrix
        BigSigma    = np.zeros((len(Data),len(Data)))
        #performing transpose of data matrix to change order
        DataT       = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
        varVect     = []
        for i in range(0,len(DataT[0])):
            vct = []
            for j in range(0,int(TrainingLen)):
                vct.append(Data[i][j])    
            varVect.append(np.var(vct))
         
        #adding variance values to diagonal elements where covariance is zero
        for j in range(len(Data)):
            BigSigma[j][j] = varVect[j]
            #scaling the variances
        if IsSynthetic == True:
            BigSigma = np.dot(3,BigSigma)
        else:
            BigSigma = np.dot(200,BigSigma)
        return BigSigma
    
#function to calculate scalar value of phi(x) for single input data row
def GetScalar(DataRow,MuRow, BigSigInv):  
        #calculating difference between Mu and input sample for a single row
        R = np.subtract(DataRow,MuRow)
        T = np.dot(BigSigInv,np.transpose(R))  
        L = np.dot(R,T)
        return L
    
#function to return final value of basis function to matrix
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
        phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
        return phi_x
    
#function to generate design matrix of 10 basis functions
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
        DataT = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
        #defining the size of design matrix
        PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
        #performing inversion of variance matrix 
        BigSigInv = np.linalg.inv(BigSigma)
        for  C in range(0,len(MuMatrix)):
            for R in range(0,int(TrainingLen)):
                #inputting values of design matrix using gaussian basis function
                PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
        return PHI
    
#function to output scalar value of Moore-Penrose psuedo inverse matrix
def GetWeightsClosedForm(PHI, T, Lambda):
        Lambda_I = np.identity(len(PHI[0]))
        for i in range(0,len(PHI[0])):
            Lambda_I[i][i] = Lambda
        PHI_T       = np.transpose(PHI)
        PHI_SQR     = np.dot(PHI_T,PHI)
        PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
        PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
        INTER       = np.dot(PHI_SQR_INV, PHI_T)
        W           = np.dot(INTER, T)
        return W
    
#array for storing ERMS and accuracy values of the model
ErmsArr = []
AccuracyArr = []
    
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
#finding cluster centres woth respect to all features using k-means clustering
Mu = kmeans.cluster_centers_
#function for generating covariance matrix
BigSigma= GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
#function call to generate design matrix for traning set
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
#function call to get final weights to compare with target vector values
W= GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))
#generating design matrix for test set 
TEST_PHI= GetPhiMatrix(TestData, Mu, BigSigma, 100) 
#generating design matrix for validation set
VAL_PHI= GetPhiMatrix(ValData, Mu, BigSigma, 100)
    
print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)
    
#function for calculating projected targets by using basis function row wise with subsequent weights
def GetValTest(VAL_PHI,W):
        Y = np.dot(W,np.transpose(VAL_PHI))
        return Y
    
#function for calculating root mean square by evaluating target - projected values
def GetErms(VAL_TEST_OUT,ValDataAct):
        sum = 0.0
        accuracy = 0.0
        counter = 0
        for i in range (0,len(VAL_TEST_OUT)):
            sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
            #checking if projected output value is equau to target value
            if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
                #if both values are equal , increment counter
                counter+=1
        #calculating accuracy percentage 
        accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
        return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))
#generating training target values 
TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
#generating validation target values 
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
#generating test target values 
TEST_OUT = GetValTest(TEST_PHI,W)
    
TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = %s \nLambda = %s" %(M,C_Lambda))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
'''********gradient solution **********'''
#generating inital random weights
W_Now        = np.dot(220, W)
#value of regularizer lambda
La           = 2
#setting of learning rate for SGD model
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []
    
for i in range(0,400):
        #calculating all paramters of error function and updated weights for iterations
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        #updating the value of weight after adding change in delta 
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        #calculatig training output value using updated weights and input design matrix
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        
        #calculatig validation output value using updated weights and input design matrix
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        
        #calculatig test output value using updated weights and input design matrix
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
print ('----------Gradient Descent Solution--------------------')
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))



