import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from IPython.display import display, HTML, clear_output
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def pretty_print(df):
    """Pretty print the dataframe

    Parameters
    ----------
    df : df
        Data Frame

    Returns
    ----------
    Display Object
        Pretty printed Data Frame
    """
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def import_and_derive_features(type_data, state):
    """Import and derive the features of each image

    Parameters
    ----------
    type_data : str
        Folder name
        
    state : str
        Normal or Pneumonia

    Returns
    ----------
    count : int
        Number of images processed
    
    dict_descriptors : dict
        Dictionary of keypoint descriptors
        
    img : array-like
        Array representation of an image
    """
    ddepth = cv.CV_16S
    kernel_size = 3
    count = 0
    dict_descriptors = {}
    path = os.path.join('chest_xrays', type_data, state)
    images = os.listdir(os.path.join('chest_xrays', type_data, state))
    for image in images:
        if image == '.ipynb_checkpoints':
            continue
        img_ = cv.imread(os.path.join(path, image))
        gray = cv.resize(img_, (256, 256))
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        gray = cv.Laplacian(gray, ddepth, ksize=kernel_size)
        gray = cv.convertScaleAbs(gray)
        sift = cv.SIFT_create()
        kp, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            continue
        dict_descriptors[image] = descriptors
        count += 1
    img = cv.drawKeypoints(gray, kp, img_, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return [count, dict_descriptors, img]

def plot_before_after():
    """Return array representation of image before
    equalizing the histogram and after.

    Returns
    ----------
    img_ : array-like
        Before
    
    gray : array-like
        After
    """
    img_ = cv.imread('original.jpeg')
    img_ = cv.resize(img_, (256, 256))
    img_ = cv.GaussianBlur(img_,(3, 3),0)
    img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(img_)
    return img_, gray

def predict(state, model, kmeans_model, scaler_k, scaler_model):
    """Predict the outcome of a single image from the test set

    Parameters
    ----------
    state : str
        Normal or Pneumonia
        
    model : function
        Best model
        
    kmeans_model : function
        Fitted kmeans-clustering model
    
    scaler_k : function
        Fitted StandardScaler for kmeans-clustering
        
    scaler_model : function
        Fitted StandardScaler for model scoring

    Returns
    ----------
    str
        Prediction
    
    img_ : array-like
        Before Processing
        
    img : array-like
        After Processing
    """
    ddepth = cv.CV_16S
    kernel_size = 3
    path = os.path.join('chest_xrays', 'test', state)
    images = os.listdir(os.path.join('chest_xrays', 'test', state))
    if state == 'normal':
        rand_file = random.randint(1, 234)
    elif state == 'pneumonia':
        rand_file = random.randint(1, 390)
    else:
        rand_file = random.randint(1, 5)
    for image in images:
        if rand_file == 1:
            print(image)
            pass
        else:
            rand_file -= 1
            continue
        if image == '.ipynb_checkpoints':
            continue
        img_ = cv.imread(os.path.join(path, image))
        gray = cv.resize(img_, (256, 256))
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        gray = cv.Laplacian(gray, ddepth, ksize=kernel_size)
        gray = cv.convertScaleAbs(gray)
        sift = cv.SIFT_create()
        kp, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            continue
        break
    img = cv.drawKeypoints(gray, kp, img_, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    descriptors = scaler_k.transform(descriptors)
    kmeans_cluster = kmeans_model.predict(descriptors)
    bovw = np.array([[0 for i in range(7)]])
    for cluster in kmeans_cluster:
        bovw[0][cluster] += 1
    bovw = scaler_model.transform(bovw)
    prediction = model.predict(bovw)
    if prediction == 0:
        return ['normal', img_, img]
    else:
        return ['pneumonia', img_, img]


def prepare_vstack(list_desc):
    """Stack the arrays vertically

    Parameters
    ----------
    list_desc : list
        List of keypoint descriptors

    Returns
    ----------
    vStack : array-like
        Array of stacked keypoint descriptors
    """
    vStack = np.array(list_desc[0])
    for row in list_desc[1:]:
        vStack = np.vstack((vStack, row))
    return vStack.copy()


def cluster(vStack, n_clusters):
    """Get kmeans-cluster

    Parameters
    ----------
    vStack : array-like
        Array of stacked keypoint descriptors
    
    n_clusters : int
        Number of kmeans-clusters

    Returns
    ----------
    kmeans_model : function
        Fitted kmeans-clustering model
        
    kmeans_cluster : array-like
        Predicted clusters for each keypoint descriptor
    """
    kmeans_model = KMeans(n_clusters, random_state=0, n_init='auto')
    kmeans_cluster = kmeans_model.fit_predict(vStack)
    return kmeans_model, kmeans_cluster


def develop_vocabulary(n_images, list_desc, n_clusters, kmeans_clusters):
    """Develop bag-of-visual-words representation for each image

    Parameters
    ----------
    n_images : int
        Number of images
        
    list_desc : list
        List of keypoint descriptors
        
    n_clusters : int
        Number of kmeans-clusters
        
    kmeans_clusters : array-like
        Predicted clusters for each keypoint descriptor

    Returns
    ----------
    bovw_hist : array-like
        Bag-of-visual-words representation of each image
    """
    bovw_hist = np.array([np.zeros(n_clusters) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        len_desc = len(list_desc[i])
        for j in range(len_desc):
            idx = kmeans_clusters[old_count + j]
            bovw_hist[i][idx] += 1
        old_count += 1
    return bovw_hist


def standardize(data, scaler=None):
    """Standardize data with StandardScaler

    Parameters
    ----------
    data : array-like
        Array
        
    scaler : function
        Fitted scaler

    Returns
    ----------
    data : array-like
        Scaled array
        
    scaler : function
        Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    else:
        data = scaler.fit_transform(data)
        return data, scaler


def get_best_params_svc(training_X, training_y, val_X, val_y, kernel):
    """Get best parameters for SVC based on given kernel

    Parameters
    ----------
    training_X : array-like
        Training X
        
    training_y : array-like
        Training y
        
    val_X : array-like
        Validation X
        
    Val-y : array-like
        Validation y
        
    kernel : str
        Kernel

    Returns
    ----------
    best_C : int or float
        Best C
        
    best_train : float
        Best training accuracy
    
    best_val : float
        Best validation accuracy
    """
    C_list = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10, 15, 20, 100, 300, 1000, 5000]
    train_accuracy = np.array([])
    val_accuracy = np.array([])
    for C in C_list:
        training_X, scaler = standardize(training_X)
        val_X = scaler.transform(val_X)
        svc = SVC(C=C, kernel=kernel, random_state=0)
        svc.fit(training_X, training_y)
        train_accuracy = np.append(train_accuracy, svc.score(training_X, training_y))
        val_accuracy = np.append(val_accuracy, svc.score(val_X, val_y))
    best_C = C_list[val_accuracy.argmax()]
    best_train = train_accuracy[val_accuracy.argmax()]
    best_val = val_accuracy.max()
    return best_C, best_train, best_val


def get_best_params_lr(training_X, training_y, val_X, val_y):
    """Get best parameters for Logistic Regression

    Parameters
    ----------
    training_X : array-like
        Training X
        
    training_y : array-like
        Training y
        
    val_X : array-like
        Validation X
        
    Val-y : array-like
        Validation y

    Returns
    ----------
    best_C : int or float
        Best C
        
    best_train : float
        Best training accuracy
    
    best_val : float
        Best validation accuracy
    """
    C_list = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.75, 1, 1.5, 3, 5, 10, 15, 20, 100, 300, 1000, 5000]
    train_accuracy = np.array([])
    val_accuracy = np.array([])
    for C in C_list:
        training_X, scaler = standardize(training_X)
        val_X = scaler.transform(val_X)
        lr = LogisticRegression(C=C, random_state=0)
        lr.fit(training_X, training_y)
        train_accuracy = np.append(train_accuracy, lr.score(training_X, training_y))
        val_accuracy = np.append(val_accuracy, lr.score(val_X, val_y))
    best_C = C_list[val_accuracy.argmax()]
    best_train = train_accuracy[val_accuracy.argmax()]
    best_val = val_accuracy.max()
    return best_C, best_train, best_val


def get_best_params_knn(training_X, training_y, val_X, val_y):
    """Get best parameters for KNN

    Parameters
    ----------
    training_X : array-like
        Training X
        
    training_y : array-like
        Training y
        
    val_X : array-like
        Validation X
        
    Val-y : array-like
        Validation y
        
    kernel : str
        Kernel

    Returns
    ----------
    k : int or float
        Best C
        
    best_train : float
        Best training accuracy
    
    best_val : float
        Best validation accuracy
    """
    n_neighbor_range = range(1, 300)
    train_accuracy = np.array([])
    val_accuracy = np.array([])
    for k in n_neighbor_range:
        training_X, scaler = standardize(training_X)
        val_X = scaler.transform(val_X)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_X, training_y)
        train_accuracy = np.append(train_accuracy, knn.score(training_X, training_y))
        val_accuracy = np.append(val_accuracy, knn.score(val_X, val_y))
    best_k = n_neighbor_range[val_accuracy.argmax()]
    best_train = train_accuracy[val_accuracy.argmax()]
    best_val = val_accuracy.max()
    return best_k, best_train, best_val


def get_best_params_rfc(training_X, training_y, val_X, val_y):
    """Get best parameters for Random Forest Classifier

    Parameters
    ----------
    training_X : array-like
        Training X
        
    training_y : array-like
        Training y
        
    val_X : array-like
        Validation X
        
    Val-y : array-like
        Validation y

    Returns
    ----------
    best_depth : int
        Best max_depth
        
    best_train : float
        Best training accuracy
    
    best_val : float
        Best validation accuracy
    """
    max_depth_range = range(5, 16)
    train_accuracy = np.array([])
    val_accuracy = np.array([])
    for depth in max_depth_range:
        training_X, scaler = standardize(training_X)
        val_X = scaler.transform(val_X)
        rfc = RandomForestClassifier(max_depth=depth, random_state=0)
        rfc.fit(training_X, training_y)
        train_accuracy = np.append(train_accuracy, rfc.score(training_X, training_y))
        val_accuracy = np.append(val_accuracy, rfc.score(val_X, val_y))
    best_depth = max_depth_range[val_accuracy.argmax()]
    best_train = train_accuracy[val_accuracy.argmax()]
    best_val = val_accuracy.max()
    return best_depth, best_train, best_val


def get_best_params_gbm(training_X, training_y, val_X, val_y):
    """Get best parameters for Gradient Boosting Classifier

    Parameters
    ----------
    training_X : array-like
        Training X
        
    training_y : array-like
        Training y
        
    val_X : array-like
        Validation X
        
    Val-y : array-like
        Validation y

    Returns
    ----------
    lst_params : list
        List containing max_depth and learning rate
        
    best_train : float
        Best training accuracy
    
    best_val : float
        Best validation accuracy
    """
    max_depth_range = range(5, 16)
    learning_rate_range = [0.1, 0.5, 1.]
    train_accuracy = np.array([])
    val_accuracy = np.array([])
    for learning_rate in learning_rate_range:
        train_acc_stack = np.array([])
        val_acc_stack = np.array([])
        for depth in max_depth_range:
            training_X, scaler = standardize(training_X)
            val_X = scaler.transform(val_X)
            gbm = GradientBoostingClassifier(max_depth=depth, learning_rate=learning_rate,
                                             random_state=0)
            gbm.fit(training_X, training_y)
            train_acc_stack = np.append(train_acc_stack, gbm.score(training_X, training_y))
            val_acc_stack = np.append(val_acc_stack, gbm.score(val_X, val_y))
        if learning_rate == 0.1:
            train_accuracy = train_acc_stack
            val_accuracy = val_acc_stack
        else:
            train_accuracy = np.vstack((train_accuracy, train_acc_stack))
            val_accuracy = np.vstack((val_accuracy, val_acc_stack))
    idx = np.unravel_index(val_accuracy.argmax(), val_accuracy.shape)
    best_lr = learning_rate_range[idx[0]]
    best_depth = max_depth_range[idx[1]]
    best_train = train_accuracy[idx[0], idx[1]]
    best_val = val_accuracy.max()
    lst_params = [best_lr, best_depth]
    return lst_params, best_train, best_val


def plot_hist(bovw, n_clusters):
    """Plot vocabulary

    Parameters
    ----------
    bovw : array-like
        Frequencies of visual words
        
    n_clusters : int
        Number of clusters
    """
    x_scalar = np.arange(n_clusters)
    y_scalar = np.array([np.sum(bovw[:, i], dtype=np.int32) for i in range(n_clusters)])
    plt.figure(figsize=(15, 15))
    plt.bar(x_scalar, y_scalar, color='black')
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def mini_batch_kmeans(vStack, n_clusters):
    """Do MiniBatchKmeans

    Parameters
    ----------
    vStack : array-like
        Stacked array of keypoint descriptors
        
    n_clusters : int
        Number of clusters
        

    Returns
    ----------
    mini_kmeans : function
        Kmeans-clustering Model
        
    kmeans_cluster : array-like
        Array of clusters for each keypoint descriptor

    """
    mini_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=200, random_state=0)
    kmeans_cluster = mini_kmeans.fit_predict(vStack)
    return mini_kmeans, kmeans_cluster