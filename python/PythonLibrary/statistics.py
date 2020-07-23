'''
This is one of the basics of Tool-Collection

Basic statistics routines
'''

import numpy; import numpy.linalg

# Numpy.array data containing all sample values, return the sample variance
def SampleVariance(data):
    f=(sum(data**2)-sum(data)**2/len(data))/(len(data)-1)
    return f

# Numpy.array yapprox & y, yapprox is the fit prediction of y, return R^2 (coefficient of determination)
def RSquare(yapprox,y):
    n=len(y)
    dev=numpy.empty(n)
    for i in range(n):
        dev[i]=yapprox[i]-y[i]
    r2=1.0-sum(dev**2)/(sum(y**2)-sum(y)**2/n)
    return r2

# Numpy.array x & y, f(x) is the fit function for y, return R^2 (coefficient of determination)
def FitFunctionRSquare(f,x,y):
    n=len(x)
    dev=numpy.empty(n)
    for i in range(n):
        dev[i]=f(x[i])-y[i]
    r2=1.0-sum(dev**2)/(sum(y**2)-sum(y)**2/n)
    return r2

# Numpy.array x & y, Order order polynomial linear least square fit x-y
# Return expansion coefficient (i-th element is the coefficient for i-th order polynomial)
def PolynomialLLSF(x,y,Order):
    N=x.shape[0]
    A=numpy.empty((Order+1,Order+1))
    b=numpy.empty(Order+1)
    coefficient=numpy.empty(Order+1)
    temp=numpy.empty(2*Order+1)
    for k in range(temp.shape[0]):
        temp[k]=0.0
        for i in range(N):
            temp[k]=temp[k]+x[i]**k
    for i in range(Order+1):
        for j in range(Order+1):
            A[i,j]=temp[i+j]
    for k in range(Order+1):
        b[k]=0.0
        for i in range(N):
            b[k]=b[k]+x[i]**k*y[i]
    coefficient=numpy.linalg.solve(A,b)
    return coefficient
# Support PolynomialLLSF
def PolyFunc(x,c):
    y=0.0
    for i in range(c.shape[0]):
        y=y+c[i]*x**i
    return y

if __name__ == "__main__": print(__doc__)