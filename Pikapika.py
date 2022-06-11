#Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Lấy dữ liệu
path = "D:\\2021-2022\\Python101\\pokemon.csv"
pokedex = pd.read_csv(open(path,'r'))
#Lấy tên pokemon
names = pokedex['name']
#Dữ liệu phân loại cho trước: Dragon Pokemon = True, Mouse pokemon = False
classfication = pokedex['classfication'].to_numpy()
classfication = (classfication == 'Dragon Pokemon')
#X bao gồm các dữ liệu kháng (3 cột cuối) Numpy 3x14
X = pokedex['against_dragon'].to_numpy().reshape(14,1)
X = np.append(X,pokedex['against_fire'].to_numpy().reshape(14,1),axis=1)
X = np.append(X,pokedex['against_water'].to_numpy().reshape(14,1),axis=1)
#Giống X nhưng chỉ chứa pokemon chuột hoặc rồng
Mouse = X[:5,:]
Dragon = X[5:,:]
#Quy chuẩn hóa dữ liệu:
#Tính trung bình cộng (temp = temporarily = biến tạm thời) và SD (xem ảnh 1,2) - Phần tóm tắt trong team
X_sum = np.sum(X,axis = 0)
X_av = X_sum / 14
X_temp = (X - X_av)**2
X_temp = np.sum(X_temp,axis=0)
SD = np.sqrt(X_temp/14)
Z = (X - X_av)/SD
#Xem ảnh 3, 3-1  (.T là chuyển vị ma trận )
cov_m = np.cov(Mouse.T,bias=True)
cov_d = np.cov(Dragon.T,bias=True)
W = ((5-1)*cov_m + (9 -1)*cov_d)/(5+9-2)
#Ảnh 4
T = np.cov(X.T)
B = T - W
S = np.linalg.inv(W) * B

eig = np.linalg.eig(S)
#Trị số cần tìm
W_  = eig[0].reshape(3,1)

#hàm tính LD dựa vào trị số tìm được
def LD(weight,data):
    temp = data * weight.T
    return np.sum(temp,axis=1)
LD1 = LD(W_,Mouse) #Tính LD của pokemon chuột và rồng
LD2 = LD(W_,Dragon)
#Tìm mức giữa của 2 LD bằng cách tìm trung bình cộng của trung bình cộng của chuột và rồng
Mouse_av = np.sum(LD1,axis = 0) / 5
Dg_av = np.sum(LD2,axis = 0) / 9

#Mức phân loại đã tìm được
LD_mid = (Mouse_av + Dg_av) / 2


#Thử phân loại
#Pikachu: Mouse Pokemon	Pikachu	1	1	1
pikapika_stat = np.array([1,1,1]).reshape(3,1)
LD_pika = np.sum(LD(W_,pikapika_stat))/3

if LD_pika >= LD_mid:
    print("Pikachu là pokemon hệ rồng")
else:
     print("Pikachu là pokemon hệ chuột")
