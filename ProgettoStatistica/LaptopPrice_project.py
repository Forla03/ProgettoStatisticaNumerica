import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import math
from sklearn.model_selection import KFold


# Caricamento del dataset
data = pd.read_csv("Data/Laptop_price.csv")

##############################
#      Pre-Processing        #
##############################

# Conversione della colonna 'Brand' in variabile categorica
data['Brand'] = data['Brand'].astype("category")

# Verifica e gestione outliers per diverse colonne
def outlier_detection(column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column_name] < lower_bound) | 
                    (data[column_name] > upper_bound)]
    return outliers.shape[0]

columns_to_check = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']
for col in columns_to_check:
    num_outliers = outlier_detection(col)
    print(f'\nNumero di outliers in {col}: {num_outliers}\n')


##############################
#          EDA               #
##############################


# Costruzione della matrice di correlazione di Pearson
numeric_data = data.select_dtypes(include='number')
C = numeric_data.corr()

plt.matshow(C, vmin=-1, vmax=1)
plt.xticks(ticks=np.arange(len(numeric_data.columns)), 
           labels=numeric_data.columns, rotation=60, ha='right')
plt.yticks(ticks=np.arange(len(numeric_data.columns)), 
           labels=numeric_data.columns)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
plt.title("Correlation Matrix")
plt.colorbar()
plt.show()


#Seleziono i 3 valori più elevati di correlazione

np.fill_diagonal(C.values, np.nan)

correlation_series = C.unstack().sort_values(ascending=False)

correlation_series = correlation_series.drop_duplicates()

# Seleziono i primi 3 valori
top_3_correlations = correlation_series.head(3)

# Stampa i risultati
print("\nI 3 indici di correlazione più elevati sono:")
print(top_3_correlations)

#Visualizzo i grafici delle variabili con indice di corr 
#più elevato

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x="Storage_Capacity", y="Price",
hue="Brand")
plt.show();

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x="RAM_Size", y="Price",
hue="Brand")
plt.show();

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x="Storage_Capacity", y="Weight",
hue="Brand")
plt.show();


##############################
#      Splitting             #
##############################

# Suddivisione del dataset in train, validation e test set
train_val_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=42)  

# La dimensione di train set è 60% gil  altri 20%

##############################
#      Regressione Lineare   #
##############################

print('\nRegressione lineare')

def linear_regression_analysis(X, y, title):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    rmse_percent = rmse / data['Price'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Punti reali')
    plt.plot(X, y_pred, color='red', label='Retta di regressione')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()
    
    print(f'Coefficiente di determinazione (r^2): {r2}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Percentuale rispetto alla media della variabile dipendente: {rmse_percent}')
    
    # Analisi dei residui
    residui = y - y_pred
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residui, kde=True, color='blue')
    plt.title('Distribuzione dei Residui')
    plt.xlabel('Residui')
    plt.ylabel('Frequenza')

    plt.subplot(1, 2, 2)
    stats.probplot(residui.flatten(), dist="norm", plot=plt)
    plt.title('Grafico Q-Q dei Residui')
    plt.show()

    # Test di normalità (Shapiro-Wilk)
    shapiro_test = stats.shapiro(residui.flatten())
    print(f'Test di Shapiro-Wilk per la normalità dei residui: {shapiro_test}\n')

# Analisi 1: Storage_Capacity vs Price
X = data['Storage_Capacity'].values.reshape(-1, 1)
y = data['Price'].values.reshape(-1, 1)
linear_regression_analysis(X, y, 'Regressione Lineare tra Storage_Capacity e Price')

# Analisi 2: RAM_Size vs Price
X = data['RAM_Size'].values.reshape(-1, 1)
y = data['Price'].values.reshape(-1, 1)
linear_regression_analysis(X, y, 'Regressione Lineare tra RAM_Size e Price')

##############################
#   Addestramento del modello #
##############################

# Funzione per addestramento e valutazione di modelli di classificazione
def train_evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"\nAccuracy {model_name} su validation set:", accuracy_score(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))
    return y_pred

# Selezione delle features e della variabile target
X_train = train_set[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
y_train = (train_set['Price'] > train_set['Price'].median()).astype(int)
X_val = val_set[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
y_val = (val_set['Price'] > val_set['Price'].median()).astype(int)
X_test = test_set[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
y_test = (test_set['Price'] > test_set['Price'].median()).astype(int)

# Standardizzazione delle features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Regressione Logistica
log_reg = LogisticRegression()
y_pred_log_reg = train_evaluate_model(log_reg, X_train, y_train, X_val, y_val, "Regressione Logistica")

# SVM con diversi kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    y_pred_svm = train_evaluate_model(svm_model, X_train, y_train, X_val, y_val, f"SVM ({kernel} kernel)")


##############################
# Hyperparameter Tuning      #
##############################

# Definizione del K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# SVM Tuning
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
    
}
svm = SVC()
svm_grid = GridSearchCV(svm, svm_params, cv=kf, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Migliori parametri per SVM:", svm_grid.best_params_)
print("Miglior score con cross-validation:", svm_grid.best_score_)

# Valutazione sui set di validazione

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_val)

print("\nAccuracy migliore SVM su validation set:", accuracy_score(y_val, y_pred_svm))
print("Confusion Matrix (SVM):\n", confusion_matrix(y_val, y_pred_svm))
print("Classification Report (SVM):\n", classification_report(y_val, y_pred_svm))


######################################
# Valutazione delle performance      #
######################################

# Predizione sul test set
y_pred_test = best_svm.predict(X_test)

# Valutazione delle performance sul test set
print("\nAccuracy su test set:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report (Test Set):\n", classification_report(y_test, y_pred_test))

#Creazione grafico di dispersione per visualizzare i risultato
#Creazione del DataFrame per la visualizzazione
test_set_visualization = pd.DataFrame(X_test, columns=['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight'])
test_set_visualization['Price'] = test_set['Price'].values
test_set_visualization['True_Label'] = y_test
test_set_visualization['Predicted_Label'] = y_pred_test
# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Storage_Capacity', y='Price', 
                hue='Predicted_Label', 
                palette='coolwarm', 
                data=test_set_visualization)

plt.title('Visualizzazione della Classificazione del Test Set con SVM')
plt.xlabel('Storage capacity')
plt.ylabel('Price')
plt.show()

#Il modello ha un'ottima accuratezza complessiva (96,5%) e 
#mostra una forte capacità di distinguere tra le due classi, soprattutto per 
#la classe 1, con una precisione del 100%.
#Anche se le prestazioni sono eccellenti, c'è una leggera tendenza a 
#classificare erroneamente alcuni campioni della classe 1 come classe 0 
#(7 falsi positivi). 
#Complessivamente, il modello è molto performante, e le sue predizioni possono
#essere considerate affidabili.

######################################
#        Studio statistico           #
######################################

k = 10  # Numero di iterazioni, non aumento che python coi cicli è lento
metrics = []

for i in range(k):
    # Re-suddivisione del dataset in train e validation set
    # con parametro random_state = i
    #Lascio i per riproducibilità dell'esperimento
    train_val_set, test_set = train_test_split(data, test_size=0.2, random_state=i)
    train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=i)

    # Selezione delle features e della variabile target
    X_train = train_set[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
    y_train = (train_set['Price'] > train_set['Price'].median()).astype(int)
    X_val = val_set[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
    y_val = (val_set['Price'] > val_set['Price'].median()).astype(int)

    # Addestramento del modello
    best_svm.fit(X_train, y_train)
    y_pred = best_svm.predict(X_val)
    
    # Calcolo delle metriche
    mse = mean_squared_error(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    metrics.append({
        'MSE': mse,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

for i, metric in enumerate(metrics):
    print(f"Iteration {i+1}: {metric}")
    
metrics_df = pd.DataFrame(metrics)

# Calcolo statistiche descrittive
statistics = metrics_df.describe()

print("\nDescriptive Statistics of Metrics:")
print(statistics)

# Grafici per le metriche
# Plot per i boxplot
plt.figure(figsize=(18, 8))

# Boxplots
for i, column in enumerate(metrics_df.columns):
    plt.subplot(2, 3, i + 1)
    plt.boxplot(metrics_df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

# Plot per gli istogrammi
plt.figure(figsize=(18, 10))

# Istogrammi
for i, column in enumerate(metrics_df.columns):
    plt.subplot(2, 3, i + 1)
    metrics_df[column].hist(bins=10, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Statstica inferenziale

mse_values = [m['MSE'] for m in metrics]
accuracy_values = [m['Accuracy'] for m in metrics]
precision_values = [m['Precision'] for m in metrics]
recall_values = [m['Recall'] for m in metrics]
f1_values = [m['F1 Score'] for m in metrics]

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # errore standard della media
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)  # margine d'errore
    return mean, mean - h, mean + h

# Calcolo della media e intervallo di confidenza per ogni metrica
mse_mean, mse_lower, mse_upper = calculate_confidence_interval(mse_values)
accuracy_mean, accuracy_lower, accuracy_upper = calculate_confidence_interval(accuracy_values)
precision_mean, precision_lower, precision_upper = calculate_confidence_interval(precision_values)
recall_mean, recall_lower, recall_upper = calculate_confidence_interval(recall_values)
f1_mean, f1_lower, f1_upper = calculate_confidence_interval(f1_values)

# Risultati
print(f"MSE: media = {mse_mean:.4f}, IC 95% = [{mse_lower:.4f}, {mse_upper:.4f}]")
print(f"Accuracy: media = {accuracy_mean:.4f}, IC 95% = [{accuracy_lower:.4f}, {accuracy_upper:.4f}]")
print(f"Precision: media = {precision_mean:.4f}, IC 95% = [{precision_lower:.4f}, {precision_upper:.4f}]")
print(f"Recall: media = {recall_mean:.4f}, IC 95% = [{recall_lower:.4f}, {recall_upper:.4f}]")
print(f"F1 Score: media = {f1_mean:.4f}, IC 95% = [{f1_lower:.4f}, {f1_upper:.4f}]")