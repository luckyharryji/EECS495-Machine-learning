def load_data():
    data = matrix(genfromtxt('breast_cancer_data.csv', delimiter=','))
    X = asarray(data[:,0:8])
    y = asarray(data[:,8])
    y.shape = (size(y),1)
    return (X,y)

def sigmoid(x):
    return 1/(1+exp(-x))

def gradient_soft(x,y,w):
    t=-y*dot(x.T,w)
    r=sigmoid(t)
    z=y*r
    gradient=-dot(x,z)
    return gradient

def hessian_soft(X,y,w):
    sum=[[0] for i in range(9)]
    for k in range(len(y)):
        at=X.T[k].reshape(1,9)
        a=X.T[k].reshape(9,1)
        sum=sum+sigmoid(-y[k]*dot(at,w))*(1-sigmoid(-y[k]*dot(at,w)))*(a*at)
    return sum


def hessian_square(X,y,w):
    total=[[0] for i in range(9)]
    for p in range(len(y)):
        if maximum(0,1-y[p]*dot(X.T[p],w))>0:
            at=X.T[p].reshape(1,9)
            a=X.T[p].reshape(9,1)
            total = total+a*at
    return 2*total

def square_grad(x,y,w):
    return -2*dot(x,maximum(0,1-y*dot(x.T,w))*y)


def newton_soft(X,y):
    steps=[]
    num_count=[]
    temp = shape(X)
    w = asarray([[0.001] for i in range(9)])
    w.reshape(9,1)

    temp = ones((temp[0],1))

    X = concatenate((temp,X),1)
    X = X.T
    grad = 1
    k = 1
    max_its = 1000
    while linalg.norm(grad) > 10**(-5) and k <= max_its:
        grad = gradient_soft(X,y,w)
        hess = linalg.inv(matrix(hessian_soft(X,y,w)))
        mis_count = 0
        for p in range(len(y)):
            if maximum(0,-y[p]*dot(X.T[p],w))>0:
                mis_count+=1
        steps.append(k)
        num_count.append(mis_count)
        w= w - asarray(hess*grad)
        k+=1
    return steps,num_count


def newton_square(X,y):
    temp = shape(X)

    steps = []
    num_count=[]
    w = asarray([[0.001] for i in range(9)])
    w.reshape(9,1)
    temp = ones((temp[0],1))

    X = concatenate((temp,X),1)
    X = X.T
    grad = 1
    k = 1
    max_its = 1000
    while linalg.norm(grad) > 10**(-8) and k <= max_its:
        grad = square_grad(X,y,w)
        hess = linalg.inv(matrix(hessian_square(X,y,w)))
        mis_count = 0
        for p in range(len(y)):
            if maximum(0,-y[p]*dot(X.T[p],w))>0:
                mis_count+=1
        steps.append(k)
        num_count.append(mis_count)
        w= w - asarray(hess*grad)
        k+=1
    return steps, num_count
