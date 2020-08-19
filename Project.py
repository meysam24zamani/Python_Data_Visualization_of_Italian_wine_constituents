from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def pca_3D_visualisation():
    wine=datasets.load_wine()
    pca = PCA(n_components=3)
    wine_pca=pca.fit_transform(wine.data)
    x=[]
    y=[]
    z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_pca:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='o',cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def pca_2D_visualisation():
    wine=datasets.load_wine()
    pca = PCA(n_components=3)
    wine_pca=pca.fit_transform(wine.data, wine.target)
    x=[]
    y=[]
    #z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_pca:
        x.append(i[0])
        y.append(i[1])
        #z.append(i[2])
    plt.scatter(x, y, c=label, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()




def get_best_classification(tree_param, kneig_param):
    wine = datasets.load_wine()
    tree=DecisionTreeClassifier()
    kneig=KNeighborsClassifier()
    grid_tree=GridSearchCV(tree, tree_param, cv=10)
    grid_kneig=GridSearchCV(kneig, kneig_param, cv=10)
    grid_tree.fit(wine.data, wine.target)
    grid_kneig.fit(wine.data, wine.target)
    results={'tree best param':grid_tree.best_params_, 'tree best score':grid_tree.best_score_, 'kneig best param': grid_kneig.best_params_, 'kneig best score': grid_kneig.best_score_}
    return(results)

def scores_with_all_features(tree_param, kneig_param):
    wine = datasets.load_wine()
    tree = DecisionTreeClassifier()
    kneig = KNeighborsClassifier()
    grid_tree = GridSearchCV(tree, tree_param, cv=5)
    grid_kneig = GridSearchCV(kneig, kneig_param, cv=5)
    grid_tree.fit(wine.data, wine.target)
    grid_kneig.fit(wine.data, wine.target)
    tree = DecisionTreeClassifier(max_depth=grid_tree.best_params_['max_depth'])
    cross_tree=cross_val_score(tree,wine.data, wine.target, cv=5)
    kneig = KNeighborsClassifier(n_neighbors=grid_kneig.best_params_['n_neighbors'])
    cross_kneig=cross_val_score(kneig, wine.data, wine.target, cv=5)
    print(sum(cross_tree)/len(cross_tree))
    print(sum(cross_kneig) / len(cross_kneig))
    return({'tree best param':grid_tree.best_params_, 'kneig best param':grid_kneig.best_params_})



def lda_2D_visualisation():
    wine=datasets.load_wine()
    lda = LDA(n_components=3)
    wine_lda=lda.fit_transform(wine.data, wine.target)
    x=[]
    y=[]
    #z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_lda:
        x.append(i[0])
        y.append(i[1])
        #z.append(i[2])
    plt.scatter(x, y, c=label, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

#Not relevant results, I decided not to use this method in my report
def ica_3D_visualisation():
    wine=datasets.load_wine()
    ica = FastICA(n_components=3)
    wine_ica=ica.fit_transform(wine.data)
    x=[]
    y=[]
    z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_ica:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='.',cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

#Not relevant results, I decided not to use this method in my report
def ica_2D_visualisation():
    wine=datasets.load_wine()
    ica = FastICA(n_components=3)
    wine_ica=ica.fit_transform(wine.data, wine.target)
    x=[]
    y=[]
    #z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_ica:
        x.append(i[0])
        y.append(i[1])
        #z.append(i[2])
    plt.scatter(x, y, c=label, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


def ker_3D_visualisation():
    wine=datasets.load_wine()
    ker = KernelPCA(n_components=3)
    wine_ker=ker.fit_transform(wine.data)
    x=[]
    y=[]
    z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_ker:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='.',cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def ker_2D_visualisation():
    wine=datasets.load_wine()
    ker = KernelPCA(n_components=3)
    wine_ker=ker.fit_transform(wine.data, wine.target)
    x=[]
    y=[]
    #z=[]
    label=[]
    label.append(wine.target)
    colors=['red', 'green', 'blue']
    for i in wine_ker:
        x.append(i[0])
        y.append(i[1])
        #z.append(i[2])
    plt.scatter(x, y, c=label, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()





