from pylab import *
from numpy import *
import matplotlib.pyplot as plt

def obj(y):
    z = log(1 + exp(dot(y.T,y)))
    return z
def grad(y):
    z = (2*exp(dot(y.T,y))*y)/(exp(dot(y.T,y)) + 1)
    return z
def hess(y):
    z = (2*exp(dot(y.T,y))*(2*dot(y.T,y) + exp(dot(y.T,y)) + 1))/(exp(dot(y.T,y)) + 1)**2
    return z
def surrogate(y,x):
    z = obj(y) + grad(y)*(x - y) + 0.5*hess(y)*(x - y)*(x - y)

    return z


def hessian_descent(w0):

    #initializations
    grad_stop = 10**-3
    max_its = 100
    iter = 0
    grad_eval = 1
    g_path = []
    w_path = []
    w_path.append(w0)
    g_path.append(obj(w0))
    w = w0
    iter_path=[]
    iter_path.append(0)
    #main loop
    while iter <= max_its:
        #take gradient step
        grad_eval = grad(w)
        hess_eval = hess(w)
        w = w - grad_eval/hess_eval
        print "x:",w
        print "y:",log(1 + exp(dot(w.T,w)))

    #    print w
        #update containers
        w_path.append(w)
    #    print "result of g:",log(1 + exp(dot(w.T,w)))
        g_path.append(log(1 + exp(dot(w.T,w))))

        #update stopers

        iter+= 1
        iter_path.append(iter)
    #print w_path
    return w_path, g_path, iter_path

def gradient_descent(w0,alpha):
    w = w0
    g_path = []
    w_path = []
    iter_path = []
    w_path.append(w)
    g_path.append(dot(w.T,w))
    iter = 0
    iter_path.append(iter)


    # start gradient descent loop



    max_its= 100
    while iter < max_its:
        #take gradient step
        grad = 2*w
        w = w - alpha*grad
        #update path container
        #print w
        w_path.append(w)
        g_path.append(dot(w.T,w))
#        print dot(w.T,w)
        iter += 1
        iter_path.append(iter)

    #print g_path
    return (w_path,g_path,iter_path)

def main():
    w0 = array([1,1,1,1,1,1,1,1,1,1])
    alpha_1 = 0.001
    alpha_2 = 0.1
    alpha_3 = 1.001
    fig=plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    fig.suptitle('Iteration speed of Newton method')
    w_path,g_path,iter_path= hessian_descent(w0)
    #print g_path
    #print w_path
    #print g_path,iter_path
    #print len(g_path),len(iter_path)
    plt.plot(iter_path,g_path,linewidth = 1.5,color = 'red')
    #w_path,g_path,iter_path= gradient_descent(w0,alpha_3)
    #plt.plot(iter_path,g_path,linewidth = 2,color = 'red',label = r'${\alpha}_3=1.001$')
    #w_path,g_path,iter_path= gradient_descent(w0,alpha_2)
    #print g_path,iter_path
    #plt.plot(iter_path,g_path,linewidth = 2,color = 'blue',label = r'${\alpha}_2=0.1$')
    #w_path,g_path,iter_path= gradient_descent(w0,alpha_1)
    #plt.plot(iter_path,g_path,linewidth = 2,color = 'green',label = r'${\alpha}_1=0.001$')
    #plt.ylim((0,100))
    plt.legend(loc = 2)
    plt.show()
main()
