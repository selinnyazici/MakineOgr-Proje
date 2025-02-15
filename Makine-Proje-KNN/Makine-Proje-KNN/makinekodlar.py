import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


##�n i�lenmi� verimizi dataSet verisine at�yoruz(Normalizasyon, dump veriler, duplicate verileri d�zenlenmi� bir �ekilde) 
dataSet = panda.read_csv('C:/Users/bahab/OneDrive/Masa�st�/heart_normalized.csv')
## veri setinin sat�r ve s�tun say�s�n� bir tuple ;(demet) olarak d�ner.
dataSet.shape



##deep=True veri setinin ba��ms�z bir kopyas�n� olu�turur. B�ylece, dataCopy(test veri seti gibi d���n�lebilir ama model �zerindeki test de�il) �zerinde yap�lan de�i�iklikler data'y� etkilemez.
dataCopy = dataSet.copy(deep = True)

correlation = dataCopy[['RestingBP', 'RestingECG']].corr()

print("Korelasyon Matrisi:")
print(correlation)



#yaava�tan model e�itim s�re�leri
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score
#�apraz do�rulama (Cross-Validation) kullanarak her kombinasyonu test eder.
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
## roc e�ri f1 skore de��imler i�in grafik olu�turmaya sa�lar 
from sklearn.model_selection import RepeatedStratifiedKFold
##pre ve recall e�ricsi �izmek i�in kullan�l�r.
from sklearn.metrics import precision_recall_curve


## Ana veriyi de�i�tirmemek ad�na copyalanm�� verideki �znitelikler features de�erine at�ld�.
## Target �zniteliklerini listeden ��kart�yoruyz ��nk� data leakage yaratmas�n diye. 
features = dataCopy[dataCopy.columns.drop(['HeartDisease'])].values
## Modelin tahmin etmeye �al��aca�� de�er.
target = dataCopy['HeartDisease'].values
## features ve target verilerini e�itim ve test setlerine b�ler.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)
## test_size = 0.20: Verinin %20'ini test seti olarak ay�r�r, geri kalan %80'i ise e�itim seti olarak kullan�l�r.
## Veriyi b�lerken rastgelelik uygulan�r, ancak bu parametre sabit bir de�er (2) ile belirlenmi�, yani her �al��t�r�ld���nda ayn� veri b�l�m� elde edilir. Bu, tekrar edilebilirlik sa�lar.
## x_train: E�itim verisinin �znitelikleri (features) (e�itim seti).
## x_test: Test verisinin �znitelikleri (features) (test seti).
## y_train: E�itim verisinin hedef de�i�keni (HeartDisease) (e�itim seti).
## y_test: Test verisinin hedef de�i�keni (HeartDisease) (test seti).

def modelsThing(classifier):
    ## .fit sayesinde model, verilerin ve hedeflerin ili�kisini ��renmeye ba�lar.
    classifier.fit(x_train,y_train)
    ## predict sayesinde e�itilen veri tahminde bulunmaya ba�lar bizde burada e�itim verisinin �zniteliklerini de�er olarak veriyoruz.
    prediction = classifier.predict(x_test)
    

    ## �apraz do�rulama y�netimidir. Amac� veri setini verdi�imiz katman(folds) (buradaki n_splits) katsay�s�na b�ler ve birden fazla kez tekrar etmesini sa�lar(n_repeats)
    # Neden yapar dersek asl�nda bir nevi veri seti dengeli ise bize genelle�tirmesi daha iyi olan overfittingden uzak daha iyi bir sonu� vermesi i�in �al���r.  
    #random_state=1 verilmesinin sebebi katlama ve b�l�mlerin ayn� �ekilde olu�mas�n� sa�lar. Verilmez ise her �al��t�rmada farkl� bir sonu� elde edilebilr.
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 5,random_state = 1)
    
    ## Bunlar classification_report k�t�phanesinde dahil de�ildi ek olrk farkl� k�t�phane �zerinden yapt�m.
    ### Anlaml� 2 basamak i�in b�yle bir kod yaz�ld�; 3.141593 yerine 3.14 yaz�lacak '{0:.2%}
    print("Do�ruluk : ",'{0:.2%}'.format(accuracy_score(y_test,prediction)))
    
    #�apraz do�rulamna yapmaya karar verdik sebebi �apraz do�rulama da verilen metrikler ile normal classification reports lar ile 
    #e� de�er ��kacak m� veya yak�n bir de�er ��kacak m� diye ek olarak kontrol yapmam�za olanak sa�lad��� i�in
    #.mean() sebebi bize bir array d�nd�rd�kleri i�in ve bize say�sal bir veri laz�m oldugundan dolay� ortalamal�r�n� bizim i�in al�p bir de�er d�nd�r�r.
    
    print("�apraz Do�rulama(RocAuc) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("�apraz Do�rulama(Accuracy(Do�ruluk)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'accuracy').mean()))
    print("�apraz Do�rulama(F1)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'f1').mean()))
    print("�apraz Do�rulama(Precision(Kesinlik) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'precision').mean()))
    print("ROC_AUC  : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    
    # modelin s�n�fland�rma (classification) performans�n� de�erlendirmek i�in precision, recall f1-score burada hesaplan�r, Support veri setindeki �rnek say�s�d�r.
    # support genelde 0 a farkl� bir say� 1 e fark�l� bir say� verir ve toplamlar� genel �rnek say�m�z�n test setindeki verdi�imis y�zdelik ile orant�l�dr.
    print(classification_report(y_test,classifier.predict(x_test)))


#Macro Precision = (0.88 + 0.86) / 2 = 0.87
#Macro Recall = (0.85 + 0.87) / 2 = 0.86
#Macro F1-Score = (0.86 + 0.86) / 2 = 0.86

#Weighted Precision = (Precision(S�n�f 1) * �rnek say�s�(S�n�f 1) + Precision(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)
#Weighted Recall = (Recall(S�n�f 1) * �rnek say�s�(S�n�f 1) + Recall(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)
#Weighted F1-Score = (F1(S�n�f 1) * �rnek say�s�(S�n�f 1) + F1(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)

    
from sklearn.neighbors import KNeighborsClassifier
## Manhattan uzakl�k �l��t� i�in
#K katsay�s� genel olarak baya denendi. 15 ten sonra belirli bir say�ya kadar d���� ya�and� 200 ler gibi say�lara geldi�i zaman %0.10 luk gibi bir do�ruluk
#artt�, fakat bu h�z� bozaca��ndan dolay� 15 ideal say� olarak d���n�lm��t�r.
#p=1 oldugu zaman manhattan 2 oldugu zaman �klid
#manhattan oldugu i�in metri�i de manhattan a ayarl�yoruz.
classKNN= KNeighborsClassifier( n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## Manhattan uzakl�k �l��t� i�in
# weights=Kom�ular�n s�n�fland�rma �zerindeki a��rl�klar�n� belirler;
#'uniform' t�m kom�ular e�it a��rl�k,'distance' Daha yak�n kom�ular daha y�ksek a��rl�k
classKNN= KNeighborsClassifier(weights='distance', n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## �klid uzakl�k �l��t� i�in
classKNN= KNeighborsClassifier( n_neighbors = 15,p = 2,metric="euclidean")
modelsThing(classKNN)

## �klid uzakl�k �l��t� i�in
classKNN= KNeighborsClassifier(weights='distance', n_neighbors = 15,p=2,metric="euclidean")
modelsThing(classKNN)


classKNN= KNeighborsClassifier(n_neighbors = 15,metric="chebyshev")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(weights='distance',n_neighbors = 15,metric="chebyshev")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(n_neighbors = 15,metric="hamming")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(weights='distance',n_neighbors = 15,metric="hamming")
modelsThing(classKNN)


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score

# Veri setini okuma ve �n i�leme
dataSet = pd.read_csv('C:/Users/bahab/OneDrive/Masa�st�/heart_normalized.csv')
dataSet.shape

dataCopy = dataSet.copy(deep=True)

# �znitelikler ve hedef de�i�kenlerin ayr�lmas�
features = dataCopy[dataCopy.columns.drop(['HeartDisease'])].values
target = dataCopy['HeartDisease'].values

# E�itim ve test verilerinin b�l�nmesi
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=2)

# Performans �l��m fonksiyonu
def evaluate_model(classifier, model_name):
    print(f"Model: {model_name}\n")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)

    # �apraz do�rulama
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    accuracy = accuracy_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print("Do�ruluk:", '{0:.2%}'.format(accuracy))
    print("F1 Skoru:", '{0:.2%}'.format(f1))
    print("Precision:", '{0:.2%}'.format(precision))
    print("Recall:", '{0:.2%}'.format(recall))
    print("�apraz Do�rulama (ROC AUC):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()))
    print("�apraz Do�rulama (Accuracy):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='accuracy').mean()))
    print("�apraz Do�rulama (F1):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='f1').mean()))
    print("�apraz Do�rulama (Precision):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='precision').mean()))
    print("�apraz Do�rulama (Recall):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='recall').mean()))
    print("ROC AUC:", '{0:.2%}'.format(roc_auc))

    # S�n�fland�rma raporu
    print("\nClassification Report:\n")
    print(classification_report(y_test, prediction))
    print("-" * 60)

    return accuracy, roc_auc, f1, precision, recall

# Grid Search ile hiperparametre optimizasyonu
def optimize_with_grid_search(classifier, param_grid, model_name):
    print(f"Grid Search ile {model_name} Hiperparametre Optimizasyonu\n")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    grid_search.fit(x_train, y_train)

    print(f"En �yi Parametreler: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return evaluate_model(best_model, f"{model_name} (Optimizasyonlu)")

# Modellerin tan�mlanmas� ve parametre aral�klar�
classKNN = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': range(1, 200),
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['distance', 'uniform']
}

classifier_lr = LogisticRegression(random_state=0,max_iter=1000)
lr_param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.1, 1, 10]},
    {'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear', 'sag', 'saga'], 'C': [0.1, 1, 10]},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
]
classifier_dt = DecisionTreeClassifier(random_state=0)
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 20),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 10)
}

# Modellerin de�erlendirilmesi ve optimizasyonu
accuracy_knn, roc_auc_knn, f1_knn, precision_knn, recall_knn = evaluate_model(classKNN, "K-Nearest Neighbors")
accuracy_lr, roc_auc_lr, f1_lr, precision_lr, recall_lr = evaluate_model(classifier_lr, "Logistic Regression")
accuracy_dt, roc_auc_dt, f1_dt, precision_dt, recall_dt = evaluate_model(classifier_dt, "Decision Tree")

opt_accuracy_knn, opt_roc_auc_knn, opt_f1_knn, opt_precision_knn, opt_recall_knn = optimize_with_grid_search(classKNN, knn_param_grid, "K-Nearest Neighbors")
opt_accuracy_lr, opt_roc_auc_lr, opt_f1_lr, opt_precision_lr, opt_recall_lr = optimize_with_grid_search(classifier_lr, lr_param_grid, "Logistic Regression")
opt_accuracy_dt, opt_roc_auc_dt, opt_f1_dt, opt_precision_dt, opt_recall_dt = optimize_with_grid_search(classifier_dt, dt_param_grid, "Decision Tree")

# En iyi modelin se�imi
best_model_name = ""
best_metric = 0
best_metrics = {}

for model_name, metrics in {
    "K-Nearest Neighbors": (opt_accuracy_knn, opt_roc_auc_knn, opt_f1_knn, opt_precision_knn, opt_recall_knn),
    "Logistic Regression": (opt_accuracy_lr, opt_roc_auc_lr, opt_f1_lr, opt_precision_lr, opt_recall_lr),
    "Decision Tree": (opt_accuracy_dt, opt_roc_auc_dt, opt_f1_dt, opt_precision_dt, opt_recall_dt)
}.items():
    if metrics[0] > best_metric:
        best_metric = metrics[0]
        best_model_name = model_name
        best_metrics = metrics

print(f"Bu veri seti i�in en iyi model: {best_model_name}")
print(f"Do�ruluk: {best_metrics[0]:.2%}, ROC AUC: {best_metrics[1]:.2%}, F1: {best_metrics[2]:.2%}, Precision: {best_metrics[3]:.2%}, Recall: {best_metrics[4]:.2%}")
