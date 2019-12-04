import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


def normalize(f, norm, msg=""):
    if (norm):
        maxv=max(f)
        if (maxv>0):
            for i in range(len(f)):
                f[i]=(f[i]/maxv)*1.0
        else:
            print("eerman bad value ("+str(t)+"): "+str(max)+" | totalBytes="+str(totalBytes))
            if msg!="":
                print(msg)
            return None
    return f

def mult_m(m, val):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j]=m[i][j]*val


def eermanFeatures(m, norm):
    f=[]
    ms=len(m)
    totalBytes=0.0
    totalBytesFrom=0.0
    totalBytesTo=0.0
    mult_m(m, 3.2192); # convert the RBG image values back to bytes
    for i in range(ms): #get all communication from/to node0
        totalBytesFrom+=m[i][0]
        totalBytesTo+=m[0][i]
    totalBytes=totalBytesFrom+totalBytesTo
    f.append(round(totalBytes/1500.0)*1.0) #[1]
    if f[0]<=0:
        return None

    f.append((totalBytesFrom/(f[0]/2.0))*1.0)#[2]
    f.append((totalBytesTo/(f[0]/2.0))*1.0)#[3]
    f.append((totalBytes/(f[0]/2.0))*1.0)#[4]
    f.append((totalBytes/100000000.0)*1.0)#[5]
    if f[3]>0:
        f.append(f[3]-129)#[6]
    else:
        f.append(0.0)#[6]
    if f[4]>0:
        f.append(1/f[4])#[7]
    else:
        f.append(0.0)
    f=normalize(f, norm)#[7]
    return f


def soysalFeatures(m, norm):
    f=[]
    ms=len(m)
    totalBytes=0.0
    mult_m(m, 3.2192); # convert the RBG image values back to bytes
    for i in range(ms):
        totalBytes+=m[i][0]
        totalBytes+=m[0][i]
    if totalBytes<=0:
        return None
    f.append(22.0) #ssh #[1]
    f.append(round(totalBytes/1500.0)*1.0)#[2]
    f.append(totalBytes*1.0)#[3]
    f.append((totalBytes/100000000.0)*1.0)#[4]
    f.append(6.0) #tcp #[5]
    f=normalize(f, norm)

    return f


def zhangFeatures(m, norm):
    f=[]
    ms=len(m)
    totalBytesFrom=0.0
    totalBytesTo=0.0
    minv=9999999999999.0
    mult_m(m, 3.2192); # convert the RBG image values back to bytes
    for i in range(ms):
        totalBytesFrom+=m[i][0]
        totalBytesTo+=m[0][i]
        if(m[i][0]>0): # get min segment size (clitoserver)
            if(m[i][0]<minv):
                minv=m[i][0]
    if (totalBytesTo+totalBytesFrom<=0):
        return None

    f.append(22.0) #ssh
    f.append(minv*1.0)
    f.append(totalBytesTo*1.0)
    f.append(totalBytesFrom*1.0)
    f=normalize(f, norm)
    return f


def fahadFeatures(m, norm):
    f=[]
    ms=len(m)
    totalBytesTo=0
    countPush=0
    mult_m(m, 3.2192); # convert the RBG image values back to bytes
    for i in range(ms): # get all communication from node0
        totalBytesTo+=m[0][i]
        if(m[i][0]>0): # get min segment size (clitoserver)
            if(m[i][0]<1500):
                countPush+=1
    if (totalBytesTo<=0):
        return None
    f.append(22.0) #ssh
    f.append(countPush*1.0)
    f.append(totalBytesTo*1.0)
    f.append(round(totalBytesTo/1500.0)*1.0)
    f=normalize(f, norm)
    return f

def pca(m, norm):
    X = m

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    pca = decomposition.PCA(n_components=1)
    X_std_pca = pca.fit_transform(X_std)

    return X_std_pca
