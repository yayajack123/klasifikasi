from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import _pickle as pickle
import requests
import json
import numpy as np
import pandas as pd
from os import system, name


# fungsi utama dalam program
def main():

    # untuk mengambil data dari file iris.csv ke data frame menggunakan panda
    dataset = pd.read_csv('iris.csv')
    pd.DataFrame(dataset)

    # mengurangi 1 kolom terakhir
    X = dataset.iloc[:, :-1]  # data

    # menampilkan data variabel dependen
    y = dataset.iloc[:, 4]  # target

    # mengubah data spesies menjadi satuan angka 0,1,2
    label_encoder = LabelEncoder()
    dataset['species'] = label_encoder.fit_transform(dataset['species'])
    dataset['species'].unique()
    label_encoder_y = LabelEncoder()
    Y = label_encoder_y.fit_transform(y)

    # membagi data training dan testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42)

    # mengubah skala data.
    scale_X = StandardScaler()
    X_train = scale_X.fit_transform(X_train)
    X_test = scale_X.transform(X_test)

    # menu input untuk memilih algoritma klasifikasi data iris
    print('Pilih Algoritma untuk klasifikasi data iris : \n')
    print('1. Naive Bayes\n' +
          '2. K-Nearest Neighbor\n' +
          '3. Decision Tree')

    # memanggil fungsi inputNumber
    var = inputNumber("Silahkan Masukan Angka : ")
    if(var == 1):
        model = GaussianNB()
        name = 'Naive Bayes'
    elif(var == 2):
        model = KNeighborsClassifier(n_neighbors=3)
        name = 'K-Nearest Neigbhor'
    elif(var == 3):
        model = DecisionTreeClassifier()
        name = 'Decision Tree'
    else:
        system('cls')
        print('Masukan Angka sesuai pilihan.')
        main()

    # Memasukkan data training pada fungsi klasifikasi
    train = model.fit(X_train, Y_train)

    # Menentukan hasil prediksi dari x_test
    y_pred = train.predict(X_test)

    # Menentukan probabilitas hasil prediksi
    train.predict_proba(X_test)

    # Menghitung nilai akurasi dari klasifikasi
    print(classification_report(Y_test, y_pred))

    # menampilkan akurasi model
    accuracy = accuracy_score(Y_test, y_pred)*100
    print('Accuracy of ' + name + ' model is equal ' +
          str(round(accuracy, 2)) + '%.')

    # memanggil fungsi untuk menampilkan pilihan kembali ke menu utama atau keluar dari program
    back()


# fungsi validasi untuk input menu klasifikasi
def inputNumber(message):
    while True:
        try:
            userInput = int(input(message))
        except ValueError:
            print("Not an integer! Try again.")
            continue
        else:
            return userInput
            break


# fungsi untuk menu pilihan kembali ke menu awal atau keluar dari program
def back():
    var = input('Klasifikasi dengan metode lain ? [yes/no] : ')
    if var == 'yes':
        system('cls')
        main()
    elif var == 'no':
        exit(0)
    else:
        print('inputan salah!!!')
        back()


# fungsi utama yang dijalankan
main()
