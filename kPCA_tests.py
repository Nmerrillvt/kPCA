# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:15:06 2019

@author: nmerr
"""
from methods.kpca import kPCA
from methods.pca import PCA
from datasets.dataLoader import loader
from utils.utils import Seedy,param_heatmap,param_scatter
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from pyod.models.ocsvm import OCSVM

Seedy(111) #sets fixed random seed
plt.close('all')
plt.rc('font', family='serif')
data_sets = [
            'Cancer',
             'Digit0',
             'Glass',
             'Ionosphere',
             'Circles',
             'Roll',
             ]



num_search = 50
methods = ['kPCA','PCA','ParzenWindow','OCSVM']
for key in data_sets:
    
    x_train, x_val, y_val, x_test, y_test = loader(key)  

    
    models = []
    scores = []
    test_aucs = []
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    """
    kPCA
    """
    num_params = num_search
    sigmas = np.logspace(-2,2,num_params)
    maxq = 100
    if maxq > x_test.shape[0] or maxq > x_val.shape[0]:
        maxq = min(x_test.shape[0],x_val.shape[0])-5
    qs = np.linspace(1,maxq,num_params,dtype = 'int')
    
    gridsearch = np.zeros((num_params*num_params,3)) #sigma rows, q, cols
    mesh = np.zeros((num_params,num_params))
    #perform gridsearch
    run = 0
    i= 0
    for sigma in sigmas:
        j = 0
        for q in qs: 
            model = kPCA(q = q, sigma = sigma)
            model.fit(x_train)
            val_scores = model.decision_function(x_val)
            auc = metrics.roc_auc_score(y_val,val_scores)
            gridsearch[run,:]= np.asarray([sigma,q,auc])
            mesh[j,i]=auc
            run += 1
            j += 1
        i += 1
    
    best_idx = np.argmax(gridsearch[:,2])      
    val_auc = np.max(gridsearch[:,2]) 
    best_sigma = gridsearch[best_idx,0]
    best_q = gridsearch[best_idx,1]
    
    param_heatmap(methods[0],fig,axs[0][0],sigmas,qs,mesh,"sigma",'q',best_sigma,best_q,log=True)
    
    model = kPCA(q = int(best_q), sigma = best_sigma)
    model.fit(x_train) #still using model data
    test_scores = model.decision_function(x_test)
    scores.append(test_scores)
    test_auc = metrics.roc_auc_score(y_test,test_scores)
    test_aucs.append(test_auc)
    print('dataset:{',key,'} method:{',methods[0],'} best params:',  '{q:',best_q,'sigma:',round(best_sigma,4),'} val auc:', round(val_auc,4),'  test auc:',round(test_auc,4))
    models.append(model)
    
    """
    PCA
    """
    num_params = x_train.shape[1]
    qs = np.linspace(1,x_train.shape[1],num_params,dtype = 'int')
    
    gridsearch = np.zeros((num_params,2))
    
    #perform gridsearch
    run = 0
    
    
    for q in qs: 
        model = PCA(q)
        model.fit(x_train)
        val_scores = model.decision_function(x_val)
        auc = metrics.roc_auc_score(y_val,val_scores)
        gridsearch[run,:]= np.asarray([q,auc])
        run += 1
    
    best_idx = np.argmax(gridsearch[:,1])      
    val_auc = np.max(gridsearch[:,1]) 
    best_q = gridsearch[best_idx,0]
    
    param_scatter(methods[1],fig,axs[1][0],gridsearch[:,0],gridsearch[:,1],'q',log=False)
    
    if best_q == x_train.shape[1]:
        best_q -= 1 #don't produce only zeros
    
    model = PCA(q = int(best_q))
    model.fit(x_train)
    test_scores = model.decision_function(x_test)
    scores.append(test_scores)
    test_auc = metrics.roc_auc_score(y_test,test_scores)
    test_aucs.append(test_auc)
    print('dataset:{',key,'} method:{',methods[1],'} best params:', '{q:',best_q,'} val auc:', round(val_auc,4),'  test auc:',round(test_auc,4))
    models.append(model)
    """
    ParzenWindow
    """
    #equivalent to kPCA with q = 0, up to a multiplicative constant
    num_params = num_search
    sigmas = np.logspace(-2,2,num_params)
    
    gridsearch = np.zeros((num_params,2)) 
    
    #perform gridsearch
    run = 0
    for sigma in sigmas: 
        model = kPCA(q = 0,sigma = sigma)
        model.fit(x_train)
        val_scores = model.decision_function(x_val)
        auc = metrics.roc_auc_score(y_val,val_scores)
        gridsearch[run,:]= np.asarray([sigma,auc])
        run += 1
    
    best_idx = np.argmax(gridsearch[:,1])      
    val_auc = np.max(gridsearch[:,1]) 
    best_sigma = gridsearch[best_idx,0]
    
    param_scatter(methods[2],fig,axs[0][1],gridsearch[:,0],gridsearch[:,1],'sigma',log=True)
    
    
    model = kPCA(q = 0, sigma = (best_sigma))
    model.fit(x_train) #still using model data
    test_scores = model.decision_function(x_test)
    scores.append(test_scores)
    test_auc = metrics.roc_auc_score(y_test,test_scores)
    test_aucs.append(test_auc)
    print('dataset:{',key,'} method:{',methods[2],'} best params:',  '{sigma:',round(best_sigma,4),'} val auc:', round(val_auc,4),'  test auc:',round(test_auc,4))
    models.append(model)
    
    """
    OCSVM
    """
    num_params = num_search
    sigmas = np.logspace(-2,2,num_params)
    nus = np.linspace(0.01,0.99,num_params)
    
    gridsearch = np.zeros((num_params*num_params,3)) #sigma rows, q, cols
    mesh = np.zeros((num_params,num_params))
    #perform gridsearch
    run = 0
    i= 0
    for sigma in sigmas:
        j = 0
        for nu in nus: 
            gamma = 0.5/(sigma*sigma)
            model = OCSVM(nu = nu, gamma = gamma)
            model.fit(x_train)
            val_scores = model.decision_function(x_val)
            auc = metrics.roc_auc_score(y_val,val_scores)
            gridsearch[run,:]= np.asarray([sigma,nu,auc])
            mesh[j,i]=auc
            run += 1
            j += 1
        i += 1
    
    best_idx = np.argmax(gridsearch[:,2])      
    val_auc = np.max(gridsearch[:,2]) 
    best_sigma = gridsearch[best_idx,0]
    best_nu = gridsearch[best_idx,1]
    
    param_heatmap(methods[3],fig,axs[1][1],sigmas,nus,mesh,'sigma','nu',best_sigma,best_nu,log=True)
    
    gamma = 0.5/(best_sigma*best_sigma)
    model = OCSVM(nu = best_nu, gamma = gamma)
    model.fit(x_train) #still using model data
    test_scores = model.decision_function(x_test)
    scores.append(test_scores)
    test_auc = metrics.roc_auc_score(y_test,test_scores)
    test_aucs.append(test_auc)
    print('dataset:{',key,'} method:{',methods[3],'} best params:',  '{nu:',round(best_nu,4),'sigma:',round(best_sigma,4),'} val auc:', round(val_auc,4),'  test auc:',round(test_auc,4))
    models.append(model)
    
    #for parameter search fig
    plt.subplots_adjust(hspace=0.5,
                        wspace=0.5)
    st = fig.suptitle(key, fontsize="x-large")
    
    #Make Roc Curves
    fig, ax = plt.subplots(figsize=(7,7))
    colors = ['g','r','b','m']
    for i in range(len(methods)):
        fpr,tpr,_ = metrics.roc_curve(y_test,scores[i],drop_intermediate = False)
        tpr[1] = 0
        if test_aucs[i] == 0.5:
            fpr = np.linspace(0,1,100)
            tpr = fpr
        ax.plot(fpr,tpr, color = colors[i], marker='o',label = methods[i],ms=2) 
    x=np.linspace(0,1,100)
    y = x
    ax.plot(x,y,ls = "--",label='random')
    ax.set_xlim([1e-4,1])
    ax.legend()
    ax.set_title(key+' ROC')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xscale('log')
    
    #Show decision boundaries for toy datasets
    if key == "Roll" or key == "Circles":
        fig, axs = plt.subplots(2,2,figsize=(10,10))
        an_idx = np.where(y_test == 1)
        bg_idx = np.where(y_test == 0)
       
        xsteps = 50
        ysteps = 50
        xmin = np.min(x_train[:,0])*1.1
        ymin = np.min(x_train[:,1])*1.1
        xmax = np.max(x_train[:,0])*1.1
        ymax = np.max(x_train[:,1])*1.1
        
        
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,xsteps), np.linspace(ymin,ymax,ysteps))
        grid_dims = xx.shape
        xx_f = np.expand_dims(xx.flatten(),axis=1)
        yy_f = np.expand_dims(yy.flatten(),axis=1)
        grid_list = np.concatenate((xx_f,yy_f),axis = 1)
        
        
        axs_ = np.reshape(axs,len(models))
        for i in range(len(models)):
            axs_[i].scatter(x_train[:,0], x_train[:,1],s=10,label = None, color = 'k',marker='s', edgecolors='k',alpha = 0.8)
            axs_[i].scatter(x_test[bg_idx,0], x_test[bg_idx,1],s=10,label = None, color = 'b',marker='s', edgecolors='b',alpha = 0.5)
            axs_[i].scatter(x_test[an_idx,0],x_test[an_idx,1], s=10,label = None, color = 'r',marker='s', edgecolors='r',alpha = 0.5)
            
            grid_errs = np.reshape(np.expand_dims(models[i].decision_function(grid_list),axis=1),grid_dims)
            
            bgval_idx = np.where(y_val == 0)
            scores = models[i].decision_function(x_val)
            fixed = np.max(scores[bgval_idx])
            cntr1 = axs_[i].contour(xx,yy,grid_errs,[fixed],colors=colors[i])
            #axs_[i].axis('off')
            axs_[i].set_xlim(xmin,xmax)
            axs_[i].set_ylim(ymin,ymax)
            axs_[i].set_title(methods[i])
            plt.subplots_adjust(wspace=0.25, hspace=0.25)
        
