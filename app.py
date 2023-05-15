import numpy as np    
from numpy import nan
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, precision_score,recall_score,cohen_kappa_score,roc_curve,auc 
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

 
#proje başarısı makine öğrenme süreci
df = pd.read_json('personel_notlari.json')
df_proje_basari_not=pd.read_json('proje_notlari.json')

proje_basari_notu=df_proje_basari_not['proje_notu'].tolist()

df_veri_yapi_bilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Veri yapılarını bilmesi') & (df['proje_kategori_id']==1)]
df_veri_analizi_yapabilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Veri analizi yapabilmesi') & (df['proje_kategori_id']==1)]
df_veri_ent_yapabilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Veri entegrasyonu yapabilmesi') & (df['proje_kategori_id']==1)]
df_sistem_analizi_bilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Sistem analizi bilmesi') & (df['proje_kategori_id']==1)]
df_sistem_tasarimi_bilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Sistem tasarımı bilmesi') & (df['proje_kategori_id']==1)]
df_donanim_ihtiyac_rapor_hazirlayabilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Donanım ihtiyaç raporu hazırlayabilmesi') & (df['proje_kategori_id']==1)]
df_proforma_fatura_toplayabilmesi=df[(df['alt_bilesen_olcut_adi'] == 'Proforma fatura toplanabilmesi') & (df['proje_kategori_id']==1)]
df_donanim_kontrol=df[(df['alt_bilesen_olcut_adi'] == 'Donanım kontrol') & (df['proje_kategori_id']==1)]
df_donanim_teslim_alinmasi=df[(df['alt_bilesen_olcut_adi'] == 'Donanım teslim alınması') & (df['proje_kategori_id']==1)]
df_birlesirme_ve_kurulma_islemleri=df[(df['alt_bilesen_olcut_adi'] == 'Donanım birleştirme ve kurulma işlemleri') & (df['proje_kategori_id']==1)]
df_sunucu_kurulum_bilgisi=df[(df['alt_bilesen_olcut_adi'] == 'Sunucu kurulum bilgisi') & (df['proje_kategori_id']==1)]
df_meksis_veri_erisim_yetkisi=df[(df['alt_bilesen_olcut_adi'] == 'MEK-SİS veriye erişim yetkisi olması') & (df['proje_kategori_id']==1)]
df_yazilim_bilgisi_olmasi=df[(df['alt_bilesen_olcut_adi'] == 'Yazılım bilgisi olması') & (df['proje_kategori_id']==1)]

# iş paketi 1
kriter1=df_veri_yapi_bilmesi['degerlendirme_notu'].tolist()
kriter2=df_veri_analizi_yapabilmesi['degerlendirme_notu'].tolist()
kriter3=df_veri_ent_yapabilmesi['degerlendirme_notu'].tolist()
kriter4=df_sistem_analizi_bilmesi['degerlendirme_notu'].tolist()
kriter5=df_sistem_tasarimi_bilmesi['degerlendirme_notu'].tolist()
# iş paketi 2
kriter6=df_donanim_ihtiyac_rapor_hazirlayabilmesi['degerlendirme_notu'].tolist()
kriter7=df_proforma_fatura_toplayabilmesi['degerlendirme_notu'].tolist()
kriter8=df_donanim_kontrol['degerlendirme_notu'].tolist()
kriter9=df_donanim_teslim_alinmasi['degerlendirme_notu'].tolist()
kriter10=df_birlesirme_ve_kurulma_islemleri['degerlendirme_notu'].tolist()
# iş paketi 3
kriter11=df_sunucu_kurulum_bilgisi['degerlendirme_notu'].tolist()
kriter12=df_meksis_veri_erisim_yetkisi['degerlendirme_notu'].tolist()
kriter13=df_yazilim_bilgisi_olmasi['degerlendirme_notu'].tolist()

ip1=list()
ip2=list()
ip3=list()

temp_satir1=list()
temp_satir2=list()
temp_satir3=list()

np_temp_satir1 = np.asarray(temp_satir1)
np_temp_satir2 = np.asarray(temp_satir2)
np_temp_satir3 = np.asarray(temp_satir3)

for i in range(0,len(kriter1),3):
    temp_satir1.append(kriter1[i])
    temp_satir1.append(kriter2[i])
    temp_satir1.append(kriter3[i])
    temp_satir1.append(kriter4[i])
    temp_satir1.append(kriter5[i])

    temp_satir2.append(kriter1[i+1])
    temp_satir2.append(kriter2[i+1])
    temp_satir2.append(kriter3[i+1])
    temp_satir2.append(kriter4[i+1])
    temp_satir2.append(kriter5[i+1])

    temp_satir3.append(kriter1[i+2])
    temp_satir3.append(kriter2[i+2])
    temp_satir3.append(kriter3[i+2])
    temp_satir3.append(kriter4[i+2])
    temp_satir3.append(kriter5[i+2])
    
    np_temp_satir1 = np.asarray(temp_satir1)
    np_temp_satir2 = np.asarray(temp_satir2)
    np_temp_satir3 = np.asarray(temp_satir3)

    np_temp_satir1 = np_temp_satir1.astype(int)
    np_temp_satir2 = np_temp_satir2.astype(int)
    np_temp_satir3 = np_temp_satir3.astype(int)

    a=(np_temp_satir1*np_temp_satir2).dot(np_temp_satir3)
    ip1.append(a)

    temp_satir1=list()
    temp_satir2=list()
    temp_satir3=list()
    
    np_temp_satir1 = np.array([])
    np_temp_satir2 = np.array([])
    np_temp_satir3 = np.array([])

for i in range(0,len(kriter1),3):
    temp_satir1.append(kriter6[i])
    temp_satir1.append(kriter7[i])
    temp_satir1.append(kriter8[i])
    temp_satir1.append(kriter9[i])
    temp_satir1.append(kriter10[i])

    temp_satir2.append(kriter6[i+1])
    temp_satir2.append(kriter7[i+1])
    temp_satir2.append(kriter8[i+1])
    temp_satir2.append(kriter9[i+1])
    temp_satir2.append(kriter10[i+1])

    temp_satir3.append(kriter6[i+2])
    temp_satir3.append(kriter7[i+2])
    temp_satir3.append(kriter8[i+2])
    temp_satir3.append(kriter9[i+2])
    temp_satir3.append(kriter10[i+2])
    
    np_temp_satir1 = np.asarray(temp_satir1)
    np_temp_satir2 = np.asarray(temp_satir2)
    np_temp_satir3 = np.asarray(temp_satir3)

    np_temp_satir1 = np_temp_satir1.astype(int)
    np_temp_satir2 = np_temp_satir2.astype(int)
    np_temp_satir3 = np_temp_satir3.astype(int)

    b=(np_temp_satir1*np_temp_satir2).dot(np_temp_satir3)
    ip2.append(b)

    temp_satir1=list()
    temp_satir2=list()
    temp_satir3=list()
    
    np_temp_satir1 = np.array([])
    np_temp_satir2 = np.array([])
    np_temp_satir3 = np.array([])

for i in range(0,len(kriter1),3):
    temp_satir1.append(kriter11[i])
    temp_satir1.append(kriter12[i])
    temp_satir1.append(kriter13[i])


    temp_satir2.append(kriter11[i+1])
    temp_satir2.append(kriter12[i+1])
    temp_satir2.append(kriter13[i+1])


    temp_satir3.append(kriter11[i+2])
    temp_satir3.append(kriter12[i+2])
    temp_satir3.append(kriter13[i+2])

    
    np_temp_satir1 = np.asarray(temp_satir1)
    np_temp_satir2 = np.asarray(temp_satir2)
    np_temp_satir3 = np.asarray(temp_satir3)

    np_temp_satir1 = np_temp_satir1.astype(int)
    np_temp_satir2 = np_temp_satir2.astype(int)
    np_temp_satir3 = np_temp_satir3.astype(int)

    c=(np_temp_satir1*np_temp_satir2).dot(np_temp_satir3)
    ip3.append(c)

    temp_satir1=list()
    temp_satir2=list()
    temp_satir3=list()
    
    np_temp_satir1 = np.array([])
    np_temp_satir2 = np.array([])
    np_temp_satir3 = np.array([])

np_ip1 = np.asarray(ip1)
np_ip2 = np.asarray(ip2)
np_ip3 = np.asarray(ip3)
y=np.asarray(proje_basari_notu)
X = np.vstack((np_ip1,np_ip2,np_ip3)).T

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.30,random_state =2)

params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    
}
model_gbr = ensemble.GradientBoostingRegressor(**params)
model_gbr.fit(x_train, y_train)
prediction_gbr=model_gbr.predict(x_test)
mse_gbr=mean_squared_error(y_test, prediction_gbr)
mae_gbr=mean_absolute_error(y_test, prediction_gbr)
rmse_gbr=mean_squared_error(y_test, prediction_gbr, squared=False)
mape_gbr=mean_absolute_percentage_error(y_test, prediction_gbr)
r2_gbr=r2_score(y_test, prediction_gbr)
score_gbr = model_gbr.score(x_train,y_train)
print(mse_gbr)
print(mae_gbr)
print(rmse_gbr)
print(mape_gbr)
print(r2_gbr)
print(score_gbr)

ip1_p1_notlar=list()
ip1_p2_notlar=list()
ip1_p3_notlar=list()

ip2_p1_notlar=list()
ip2_p2_notlar=list()
ip2_p3_notlar=list()

ip3_p1_notlar=list()
ip3_p2_notlar=list()
ip3_p3_notlar=list()


#proje başarı durumu sınıflandırma makine öğrenmesi süreci
proje_basari_durumu=list()
for i in range (len(proje_basari_notu)):
    if proje_basari_notu[i]>=70:
        proje_basari_durumu.append(1)
    else:
        proje_basari_durumu.append(0)

y_c=np.asarray(proje_basari_durumu)
x_train_c , x_test_c , y_train_c , y_test_c = train_test_split(X , y_c , test_size = 0.30,random_state =2)

from sklearn.ensemble import GradientBoostingClassifier
params_gbc = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    
}
model_gbc = GradientBoostingClassifier(**params_gbc)
model_gbc.fit(x_train_c, y_train_c)
prediction_gbc = model_gbc.predict(x_test_c)
gbc_acc=model_gbc.score(x_train_c, y_train_c)
gbc_f1=f1_score(y_test_c,prediction_gbc)
gbc_precision=precision_score(y_test_c,prediction_gbc)
gbc_recall=recall_score(y_test_c,prediction_gbc)
gbc_kappa=cohen_kappa_score(y_test_c,prediction_gbc)
print(gbc_acc)
print(gbc_f1)
print(gbc_precision)
print(gbc_recall)
print(gbc_kappa)

#personel not tahmini veri önişleme süreci
df = pd.read_json('personel_notlari.json')

df_ip1_notlar=df[(df['alt_bilesen_adi'] == 'Kampüs ile İlgili var olan mekansal verilerin toplanması ve analiz edilmesi') & (df['proje_kategori_id']==1)]
df_ip2_notlar=df[(df['alt_bilesen_adi'] == 'Donanımların Temini') & (df['proje_kategori_id']==1)]
df_ip3_notlar=df[(df['alt_bilesen_adi'] == 'ArcGIS ve Web Sunucunun Kurulumu') & (df['proje_kategori_id']==1)]

ip1_notlar_list=df_ip1_notlar['degerlendirme_notu'].tolist()
ip2_notlar_list=df_ip2_notlar['degerlendirme_notu'].tolist()
ip3_notlar_list=df_ip3_notlar['degerlendirme_notu'].tolist()

ip1_k1_list=list()
ip1_k2_list=list()
ip1_k3_list=list()
ip1_k4_list=list()
ip1_k5_list=list()
ip1_per_not_list=list()

ip2_k1_list=list()
ip2_k2_list=list()
ip2_k3_list=list()
ip2_k4_list=list()
ip2_k5_list=list()
ip2_per_not_list=list()

ip3_k1_list=list()
ip3_k2_list=list()
ip3_k3_list=list()
ip3_per_not_list=list()

for i in range(len(ip1_notlar_list)):
    if (i%6==0):
        ip1_k1_list.append(ip1_notlar_list[i])
    if (i%6==1):
        ip1_k2_list.append(ip1_notlar_list[i])
    if (i%6==2):
        ip1_k3_list.append(ip1_notlar_list[i])
    if (i%6==3):
        ip1_k4_list.append(ip1_notlar_list[i])
    if (i%6==4):
        ip1_k5_list.append(ip1_notlar_list[i])
    if (i%6==5):
        ip1_per_not_list.append(ip1_notlar_list[i])


for i in range(len(ip2_notlar_list)):
    if (i%6==0):
        ip2_k1_list.append(ip2_notlar_list[i])
    if (i%6==1):
        ip2_k2_list.append(ip2_notlar_list[i])
    if (i%6==2):
        ip2_k3_list.append(ip2_notlar_list[i])
    if (i%6==3):
        ip2_k4_list.append(ip2_notlar_list[i])
    if (i%6==4):
        ip2_k5_list.append(ip2_notlar_list[i])
    if (i%6==5):
        ip2_per_not_list.append(ip2_notlar_list[i])


for i in range(len(ip3_notlar_list)):
    if (i%4==0):
        ip3_k1_list.append(ip3_notlar_list[i])
    if (i%4==1):
        ip3_k2_list.append(ip3_notlar_list[i])
    if (i%4==2):
        ip3_k3_list.append(ip3_notlar_list[i])
    if (i%4==3):
        ip3_per_not_list.append(ip3_notlar_list[i])

#iş paketi 1 personel not tahmini makine öğrenmesi süreci
np_ip1_k1 = np.asarray(ip1_k1_list)
np_ip1_k2 = np.asarray(ip1_k2_list)
np_ip1_k3 = np.asarray(ip1_k3_list)
np_ip1_k4 = np.asarray(ip1_k4_list)
np_ip1_k5 = np.asarray(ip1_k5_list)

y_ip1_per_not = np.asarray(ip1_per_not_list)
X_ip1_per_not = np.vstack((np_ip1_k1, np_ip1_k2,np_ip1_k3,np_ip1_k4,np_ip1_k5)).T

x_train_ip1_per_not , x_test_ip1_per_not , y_train_ip1_per_not , y_test_ip1_per_not = train_test_split(X_ip1_per_not , y_ip1_per_not , test_size = 0.30,random_state =2)

model_rfr_ip1_per_not = RandomForestRegressor(max_depth=5, min_samples_split=3, n_estimators=24,random_state=0)
model_rfr_ip1_per_not.fit(x_train_ip1_per_not, y_train_ip1_per_not)

#iş paketi 2 personel not tahmini makine öğrenmesi süreci
np_ip2_k1 = np.asarray(ip2_k1_list)
np_ip2_k2 = np.asarray(ip2_k2_list)
np_ip2_k3 = np.asarray(ip2_k3_list)
np_ip2_k4 = np.asarray(ip2_k4_list)
np_ip2_k5 = np.asarray(ip2_k5_list)

y_ip2_per_not = np.asarray(ip2_per_not_list)
X_ip2_per_not = np.vstack((np_ip2_k1, np_ip2_k2,np_ip2_k3,np_ip2_k4,np_ip2_k5)).T

x_train_ip2_per_not , x_test_ip2_per_not , y_train_ip2_per_not , y_test_ip2_per_not = train_test_split(X_ip2_per_not , y_ip2_per_not , test_size = 0.30,random_state =2)

model_rfr_ip2_per_not = RandomForestRegressor(max_depth=5, min_samples_split=3, n_estimators=24,random_state=0)
model_rfr_ip2_per_not.fit(x_train_ip2_per_not, y_train_ip2_per_not)

#iş paketi 3 personel not tahmini makine öğrenmesi süreci
np_ip3_k1 = np.asarray(ip3_k1_list)
np_ip3_k2 = np.asarray(ip3_k2_list)
np_ip3_k3 = np.asarray(ip3_k3_list)


y_ip3_per_not = np.asarray(ip3_per_not_list)
X_ip3_per_not = np.vstack((np_ip3_k1, np_ip3_k2,np_ip3_k3)).T

x_train_ip3_per_not , x_test_ip3_per_not , y_train_ip3_per_not , y_test_ip3_per_not = train_test_split(X_ip3_per_not , y_ip3_per_not , test_size = 0.30,random_state =2)

model_rfr_ip3_per_not = RandomForestRegressor(max_depth=5, min_samples_split=3, n_estimators=24,random_state=0)
model_rfr_ip3_per_not.fit(x_train_ip3_per_not, y_train_ip3_per_not)



# Flask constructor
app = Flask(__name__,template_folder='templates', static_folder='static')  
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def index():
   return render_template("index.html")

@app.route('/sonuc', methods =["GET", "POST"])
def sonuc():
   if request.method=="POST":
      
      # formdan veri alma - başlagıç
      ip_1_per_1_k_1=request.form.get("ip_1_per_1_k_1")
      ip_1_per_1_k_2=request.form.get("ip_1_per_1_k_2")
      ip_1_per_1_k_3=request.form.get("ip_1_per_1_k_3")
      ip_1_per_1_k_4=request.form.get("ip_1_per_1_k_4")
      ip_1_per_1_k_5=request.form.get("ip_1_per_1_k_5")
      ip_1_per_2_k_1=request.form.get("ip_1_per_2_k_1")
      ip_1_per_2_k_2=request.form.get("ip_1_per_2_k_2")
      ip_1_per_2_k_3=request.form.get("ip_1_per_2_k_3")
      ip_1_per_2_k_4=request.form.get("ip_1_per_2_k_4")
      ip_1_per_2_k_5=request.form.get("ip_1_per_2_k_5")
      ip_1_per_3_k_1=request.form.get("ip_1_per_3_k_1")
      ip_1_per_3_k_2=request.form.get("ip_1_per_3_k_2")
      ip_1_per_3_k_3=request.form.get("ip_1_per_3_k_3")
      ip_1_per_3_k_4=request.form.get("ip_1_per_3_k_4")
      ip_1_per_3_k_5=request.form.get("ip_1_per_3_k_5")
      


      ip_2_per_1_k_1=request.form.get("ip_2_per_1_k_1")
      ip_2_per_1_k_2=request.form.get("ip_2_per_1_k_2")
      ip_2_per_1_k_3=request.form.get("ip_2_per_1_k_3")
      ip_2_per_1_k_4=request.form.get("ip_2_per_1_k_4")
      ip_2_per_1_k_5=request.form.get("ip_2_per_1_k_5")
      ip_2_per_2_k_1=request.form.get("ip_2_per_2_k_1")
      ip_2_per_2_k_2=request.form.get("ip_2_per_2_k_2")
      ip_2_per_2_k_3=request.form.get("ip_2_per_2_k_3")
      ip_2_per_2_k_4=request.form.get("ip_2_per_2_k_4")
      ip_2_per_2_k_5=request.form.get("ip_2_per_2_k_5")
      ip_2_per_3_k_1=request.form.get("ip_2_per_3_k_1")
      ip_2_per_3_k_2=request.form.get("ip_2_per_3_k_2")
      ip_2_per_3_k_3=request.form.get("ip_2_per_3_k_3")
      ip_2_per_3_k_4=request.form.get("ip_2_per_3_k_4")
      ip_2_per_3_k_5=request.form.get("ip_2_per_3_k_5")


      ip_3_per_1_k_1=request.form.get("ip_3_per_1_k_1")
      ip_3_per_1_k_2=request.form.get("ip_3_per_1_k_2")
      ip_3_per_1_k_3=request.form.get("ip_3_per_1_k_3")
      ip_3_per_2_k_1=request.form.get("ip_3_per_2_k_1")
      ip_3_per_2_k_2=request.form.get("ip_3_per_2_k_2")
      ip_3_per_2_k_3=request.form.get("ip_3_per_2_k_3")
      ip_3_per_3_k_1=request.form.get("ip_3_per_3_k_1")
      ip_3_per_3_k_2=request.form.get("ip_3_per_3_k_2")
      ip_3_per_3_k_3=request.form.get("ip_3_per_3_k_3")


      ip_1_per_1_ad=request.form.get("ip_1_per_1")
      ip_1_per_2_ad=request.form.get("ip_1_per_2")
      ip_1_per_3_ad=request.form.get("ip_1_per_3")
      ip_2_per_1_ad=request.form.get("ip_2_per_1")
      ip_2_per_2_ad=request.form.get("ip_2_per_2")
      ip_2_per_3_ad=request.form.get("ip_2_per_3")
      ip_3_per_1_ad=request.form.get("ip_3_per_1")
      ip_3_per_2_ad=request.form.get("ip_3_per_2")
      ip_3_per_3_ad=request.form.get("ip_3_per_3")

      # formdan veri alma - bitiş

      
      #iş paketi 1 nokta verisi üretme - başlangıç
      ip1_p1_notlar.append(int(ip_1_per_1_k_1))
      ip1_p1_notlar.append(int(ip_1_per_1_k_2))
      ip1_p1_notlar.append(int(ip_1_per_1_k_3))
      ip1_p1_notlar.append(int(ip_1_per_1_k_4))
      ip1_p1_notlar.append(int(ip_1_per_1_k_5))

      ip1_p2_notlar.append(int(ip_1_per_2_k_1))
      ip1_p2_notlar.append(int(ip_1_per_2_k_2))
      ip1_p2_notlar.append(int(ip_1_per_2_k_3))
      ip1_p2_notlar.append(int(ip_1_per_2_k_4))
      ip1_p2_notlar.append(int(ip_1_per_2_k_5))

      ip1_p3_notlar.append(int(ip_1_per_3_k_1))
      ip1_p3_notlar.append(int(ip_1_per_3_k_2))
      ip1_p3_notlar.append(int(ip_1_per_3_k_3))
      ip1_p3_notlar.append(int(ip_1_per_3_k_4))
      ip1_p3_notlar.append(int(ip_1_per_3_k_5))

      np_ip1_p1_notlar = np.asarray(ip1_p1_notlar)
      np_ip1_p2_notlar = np.asarray(ip1_p2_notlar)
      np_ip1_p3_notlar = np.asarray(ip1_p3_notlar)

      np_ip1_p1_notlar = np_ip1_p1_notlar.astype(int)
      np_ip1_p2_notlar = np_ip1_p2_notlar.astype(int)
      np_ip1_p3_notlar = np_ip1_p3_notlar.astype(int)

      ip1_form=(np_ip1_p1_notlar*np_ip1_p2_notlar).dot(np_ip1_p3_notlar)

      #iş paketi 2 nokta verisi üretme - başlangıç

      ip2_p1_notlar.append(int(ip_2_per_1_k_1))
      ip2_p1_notlar.append(int(ip_2_per_1_k_2))
      ip2_p1_notlar.append(int(ip_2_per_1_k_3))
      ip2_p1_notlar.append(int(ip_2_per_1_k_4))
      ip2_p1_notlar.append(int(ip_2_per_1_k_5))

      ip2_p2_notlar.append(int(ip_2_per_2_k_1))
      ip2_p2_notlar.append(int(ip_2_per_2_k_2))
      ip2_p2_notlar.append(int(ip_2_per_2_k_3))
      ip2_p2_notlar.append(int(ip_2_per_2_k_4))
      ip2_p2_notlar.append(int(ip_2_per_2_k_5))

      ip2_p3_notlar.append(int(ip_2_per_3_k_1))
      ip2_p3_notlar.append(int(ip_2_per_3_k_2))
      ip2_p3_notlar.append(int(ip_2_per_3_k_3))
      ip2_p3_notlar.append(int(ip_2_per_3_k_4))
      ip2_p3_notlar.append(int(ip_2_per_3_k_5))

      np_ip2_p1_notlar = np.asarray(ip2_p1_notlar)
      np_ip2_p2_notlar = np.asarray(ip2_p2_notlar)
      np_ip2_p3_notlar = np.asarray(ip2_p3_notlar)

      np_ip2_p1_notlar = np_ip2_p1_notlar.astype(int)
      np_ip2_p2_notlar = np_ip2_p2_notlar.astype(int)
      np_ip2_p3_notlar = np_ip2_p3_notlar.astype(int)

      ip2_form=(np_ip2_p1_notlar*np_ip2_p2_notlar).dot(np_ip2_p3_notlar)

      #iş paketi 3 nokta verisi üretme - başlangıç

      ip3_p1_notlar.append(int(ip_3_per_1_k_1))
      ip3_p1_notlar.append(int(ip_3_per_1_k_2))
      ip3_p1_notlar.append(int(ip_3_per_1_k_3))
      

      ip3_p2_notlar.append(int(ip_3_per_2_k_1))
      ip3_p2_notlar.append(int(ip_3_per_2_k_2))
      ip3_p2_notlar.append(int(ip_3_per_2_k_3))
      

      ip3_p3_notlar.append(int(ip_3_per_3_k_1))
      ip3_p3_notlar.append(int(ip_3_per_3_k_2))
      ip3_p3_notlar.append(int(ip_3_per_3_k_3))
    

      np_ip3_p1_notlar = np.asarray(ip3_p1_notlar)
      np_ip3_p2_notlar = np.asarray(ip3_p2_notlar)
      np_ip3_p3_notlar = np.asarray(ip3_p3_notlar)

      np_ip3_p1_notlar = np_ip3_p1_notlar.astype(int)
      np_ip3_p2_notlar = np_ip3_p2_notlar.astype(int)
      np_ip3_p3_notlar = np_ip3_p3_notlar.astype(int)

      ip3_form=(np_ip3_p1_notlar*np_ip3_p2_notlar).dot(np_ip3_p3_notlar)

      #nokta verisi üretme son


      X_formdan_gelen= np.vstack((ip1_form,ip2_form,ip3_form)).T
      prediction_gbr=model_gbr.predict(X_formdan_gelen)
      prediction_gbc=model_gbc.predict(X_formdan_gelen)

      X_ip1_p1_notlar=np.vstack((np_ip1_p1_notlar)).T
      prediction_rfr_ip1_per1_not=model_rfr_ip1_per_not.predict(X_ip1_p1_notlar)

      X_ip1_p2_notlar=np.vstack((np_ip1_p2_notlar)).T
      prediction_rfr_ip1_per2_not=model_rfr_ip1_per_not.predict(X_ip1_p2_notlar)

      X_ip1_p3_notlar=np.vstack((np_ip1_p3_notlar)).T
      prediction_rfr_ip1_per3_not=model_rfr_ip1_per_not.predict(X_ip1_p3_notlar)


      X_ip2_p1_notlar=np.vstack((np_ip2_p1_notlar)).T
      prediction_rfr_ip2_per1_not=model_rfr_ip2_per_not.predict(X_ip2_p1_notlar)

      X_ip2_p2_notlar=np.vstack((np_ip2_p2_notlar)).T
      prediction_rfr_ip2_per2_not=model_rfr_ip2_per_not.predict(X_ip2_p2_notlar)

      X_ip2_p3_notlar=np.vstack((np_ip2_p3_notlar)).T
      prediction_rfr_ip2_per3_not=model_rfr_ip2_per_not.predict(X_ip2_p3_notlar)


      X_ip3_p1_notlar=np.vstack((np_ip3_p1_notlar)).T
      prediction_rfr_ip3_per1_not=model_rfr_ip3_per_not.predict(X_ip3_p1_notlar)

      X_ip3_p2_notlar=np.vstack((np_ip3_p2_notlar)).T
      prediction_rfr_ip3_per2_not=model_rfr_ip3_per_not.predict(X_ip3_p2_notlar)

      X_ip3_p3_notlar=np.vstack((np_ip3_p3_notlar)).T
      prediction_rfr_ip3_per3_not=model_rfr_ip3_per_not.predict(X_ip3_p3_notlar)

      return render_template("sonuc.html",basari_notu=int(prediction_gbr),basari_durumu=prediction_gbc,ip1_per1_not=int(prediction_rfr_ip1_per1_not),
                                                                                                  ip1_per2_not=int(prediction_rfr_ip1_per2_not),
                                                                                                  ip1_per3_not=int(prediction_rfr_ip1_per3_not),
                                                                                                  ip2_per1_not=int(prediction_rfr_ip2_per1_not),
                                                                                                  ip2_per2_not=int(prediction_rfr_ip2_per2_not),
                                                                                                  ip2_per3_not=int(prediction_rfr_ip2_per3_not),
                                                                                                  ip3_per1_not=int(prediction_rfr_ip3_per1_not),
                                                                                                  ip3_per2_not=int(prediction_rfr_ip3_per2_not),
                                                                                                  ip3_per3_not=int(prediction_rfr_ip3_per3_not),
                                                                                                  ip1_per1_ad=ip_1_per_1_ad,
                                                                                                  ip1_per2_ad=ip_1_per_2_ad,
                                                                                                  ip1_per3_ad=ip_1_per_3_ad,
                                                                                                  ip2_per1_ad=ip_2_per_1_ad,
                                                                                                  ip2_per2_ad=ip_2_per_2_ad,
                                                                                                  ip2_per3_ad=ip_2_per_3_ad,
                                                                                                  ip3_per1_ad=ip_3_per_1_ad,
                                                                                                  ip3_per2_ad=ip_3_per_2_ad,
                                                                                                  ip3_per3_ad=ip_3_per_3_ad)
                                                                                                  
   else:
      pass

if __name__=='__main__':
   app.run()