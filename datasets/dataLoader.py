# -*- coding: utf-8 -*-
"""
datasets sourced from:


"""
import numpy as np
from numpy import random



def check_d(edge,point,thresh):
    n = edge.shape[0]
    p_exp = np.repeat(np.expand_dims(point,axis=0),n,axis=0)
    diff = edge - p_exp
    dist = np.sqrt(diff[:,0]*diff[:,0] + diff[:,1]*diff[:,1])
    if np.min(dist) > thresh:
        return True
    else:
        return False


def loader(key, directory = 'datasets/',makeVal=True):

    if key == 'Roll':
        
        n_bg_train = 400
        n_bg_test = n_bg_train
        
        n1 = n_bg_test + n_bg_train#n_background
        n2 = 300#n_anomalies
        
        data = np.zeros((n2+n1,2))
        edge = np.zeros((4*n1,2))
        labels = np.zeros((n1+n2))
        
        buffer = 0.6
        a=0
        b=0.7
        c=1.25
        width = 0.5
        loops = 2.5
    
        for i in range(n2+n1):
            thetas = np.linspace(0,loops*2*np.pi,n1)       
            if i < n1:
                
                theta = thetas[i]
                
                r = (a+b*theta)
                x = r*np.cos(theta)+(2*random.randint(0, 1)-1)*random.uniform(-width,width)
                y = r*np.sin(theta)+(2*random.randint(0, 1)-1)*random.uniform(-width,width)#

                data_i = np.array([x,y])
                data[i,:] = data_i
                labels[i] = 0

                bg_data = data[:n1]
                max_Data = np.max(bg_data)
                min_Data = np.min(bg_data) 

  
                
   
            if i > n1:
                go = True
                
                while go:
                    x = random.uniform(min_Data,max_Data)
                    y = random.uniform(min_Data,max_Data)
                    if check_d(bg_data,np.array([x,y]),buffer):
                        go = False

                data_i = np.array([x,y])
                data[i,:] = data_i
                labels[i] = 1
                
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(data[:,0], data[:,1],s=10,label = None, color = 'k',marker='s', edgecolors='k',alpha = 0.8)
        ax.scatter(edge[:,0], edge[:,1],s=10,label = None, color = 'b',marker='s', edgecolors='b',alpha = 0.5)    
        
        shuff = np.random.permutation(n1+n2)        
        data = data[shuff]
        labels = labels[shuff]

        an_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        x_train = data[bg_idx[:n_bg_train]]
        x_bg_test = data[bg_idx[n_bg_train:]]
        x_an_test = data[an_idx]
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
 
    elif key == 'Circles':
        
        n_bg_train = 600
        n_bg_test = n_bg_train
        
        n1 = n_bg_test + n_bg_train #n_background
        n2 = 600 #n_anomalies

        data = np.zeros((n2+n1,2))
        
        buffer = 0.025
        a2=0.35
        b1=0.75
        b2=0.9
        
        pi = 3.141
        Area_a = pi*(a2*a2)
        Area_b = pi*(b2*b2 - b1*b1)
        prob_in_b = Area_b/(Area_b+Area_a)
        
        for i in range(n2+n1):
            if i < n2:
                r = (a2)/2
                while (r < a2+buffer) or (r > b1-buffer and r < b2+buffer):
                    x = random.uniform(-1.2,1.2)
                    y = random.uniform(-1.2,1.2)
                    r = np.sqrt(x*x+y*y)
                data_i = np.array([x,y])            
            
            
            if i > n2:
                roll = random.uniform(0,1)
                if roll > prob_in_b:
                    r = random.uniform(0,a2)
                else:
                    r = random.uniform(b1,b2)
        
                theta = random.uniform(0,2*np.pi)
                data_i = np.array([r*np.cos(theta),r*np.sin(theta)])
            
            data[i,:] = data_i
           
            
        labels = np.zeros((n1+n2))
        labels[:n2] = 1              
        
        an_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        x_train = data[bg_idx[:n_bg_train]]
        x_bg_test = data[bg_idx[n_bg_train:]]
        x_an_test = data[an_idx]
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
        

    elif key == 'Cancer':
        """
        Quoting Hoffman:
            
        The breast-cancer data were obtained from the UCI machine-learning 
        repository [18]. These data were collected by Dr. William H. Wolberg 
        at the University of Wisconsin Hospitals in Madison [19]. The patterns
        in this data set belong to two classes: benign and malignant. 
        Each pattern consists of nine cytological characteristics such as,
        for example, the uniformity of cell size. Each of these characteristics
        is graded with an integer value from 1 to 10, with 1 being typical benign. 
        The database contains some patterns with missing attributes, these 
        patterns were removed before further processing. The remaining patterns
        were scaled to have unit variance in each dimension. To avoid numerical 
        errors because of the discrete values, a uniform noise from the interval 
        [−0.05,0.05] was added to each value. The novelty detectors were trained 
        on the ﬁrst 200 benign samples. The remaining samples were used 
        for testing: 244 benign and 239 malignant.
        """
        n_bg_train = 200
        U = 0.05
        
        contents = np.genfromtxt(directory+"cancer.csv", delimiter=',')
        data = contents[:,0:-1]
        data = data + np.random.uniform(-U,U,size = data.shape)
        std = np.repeat(np.expand_dims(np.std(data,axis=0),axis=0),data.shape[0],axis=0)
        data = (data)*1/std
        
        
        labels = contents[:,-1]
        
        an_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        x_train = data[bg_idx[:n_bg_train]]
        x_bg_test = data[bg_idx[n_bg_train:]]
        x_an_test = data[an_idx]
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
        
        
    elif key == 'Glass':
        n_bg_train = 100
        U = 0.05
        
        contents = np.genfromtxt(directory+"glass.csv", delimiter=',')
        data = contents[:,0:-1]
        data = data + np.random.uniform(-U,U,size = data.shape)
        std = np.repeat(np.expand_dims(np.std(data,axis=0),axis=0),data.shape[0],axis=0)
        data = (data)*1/std
        
        
        labels = contents[:,-1]
        
        an_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        x_train = data[bg_idx[:n_bg_train]]
        x_bg_test = data[bg_idx[n_bg_train:]]
        x_an_test = data[an_idx]
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
        
        
    elif key == 'Ionosphere':
        n_bg_train = 180
        U = 0.05
        
        contents = np.genfromtxt(directory+"ionosphere.csv", delimiter=',')
        data = contents[:,0:-1]
        data = data + np.random.uniform(-U,U,size = data.shape)
        std = np.repeat(np.expand_dims(np.std(data,axis=0),axis=0),data.shape[0],axis=0)
        data = (data)*1/std
        
        
        labels = contents[:,-1]
        
        an_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        x_train = data[bg_idx[:n_bg_train]]
        x_bg_test = data[bg_idx[n_bg_train:]]
        x_an_test = data[an_idx]
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
        
        
    elif key == 'Digit0':
        """
        Quoting Hoffman:
        Digit 0: The digits were obtained from the MNIST digit database [17].
        The original 28×28 pixels images are almost binary (see Fig. 11). 
        Thus, the digits occupy only the corners of a 784-dimensional hyper-cube. 
        To get a more continuous distribution of digits, the original images 
        were blurred and subsampled down to 8×8 pixels. The MNIST database is 
        split into training set and test set. To train the novelty detectors, 
        the ﬁrst 2000 ‘0’ digits from the training set were used. 
        For the 0/not-0 classiﬁcation task, from the test set, all 980 ‘0’ 
        digits were used together with the ﬁrst 109 samples from each other digit.
        
        """
        from scipy.ndimage.filters import gaussian_filter
        from skimage.transform import resize
        from keras.datasets import mnist
        
        background_digit = 0
        anomaly_digits = [1,2,3,4,5,6,7,8,9]
        new_size = 8
        n_bg_train = 2000
        n_bg_test = 980
        n_an_test = 109
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = np.vstack((x_train,x_test))
        labels = np.concatenate((y_train,y_test))
        
        data = data.astype('float32') / 255
        bg_idx = np.where(labels == background_digit)
         
        
        x_train = data[bg_idx][:n_bg_train]
        x_bg_test = data[bg_idx][n_bg_train:n_bg_train+n_bg_test]
        
        an_idx = []
        for d in anomaly_digits:
            idx = np.where(labels == d)
            an_idx.append(idx)
        
        an_test = []
        for d in range(len(anomaly_digits)):
            idx = an_idx[d][0][:n_an_test]
            an_test.append(data[idx])
        x_an_test = np.vstack(an_test)
        
        y_test = np.zeros((x_bg_test.shape[0]+x_an_test.shape[0]))
        y_test[x_bg_test.shape[0]:]=1
        x_test = np.vstack((x_bg_test,x_an_test))
        
        #blur and resize and flatten
        x_train = gaussian_filter(x_train, sigma=1)#sigma not specified in paper
        x_test = gaussian_filter(x_test, sigma=1)#sigma not specified in paper
        x_train = resize(x_train, (x_train.shape[0],new_size,new_size))
        x_test = resize(x_test, (x_test.shape[0],new_size,new_size))
        
        #flatten
        x_train = np.reshape(x_train, [-1, new_size*new_size])
        x_test = np.reshape(x_test, [-1, new_size*new_size])
        
    else:
        assert False, 'Use available key.'
        
    if makeVal:
        test_size = x_test.shape[0]
        idx = np.random.randint(0,test_size, int(test_size/2))
        x_val = x_test[idx]
        y_val = y_test[idx]
        x_test = x_test[~idx]
        y_test = y_test[~idx]
    
    return x_train, x_val, y_val, x_test, y_test

# =============================================================================




