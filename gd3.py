import random

k = 3
iterations = 299
l = 1

def M(x, m, b):
    return m * x + b

class dM:
    def dm(x, m, b):        
        return x
    def db(x, m, b):
        return 1

def loss(x, y, M, m, b):
    return (y - M(x, m, b)) ** 2

class dloss:
    def dm(x, y, M, m, b):
        return 2 * (y - M(x, m, b)) * (-dM.dm(x, m, b))
    def db(x, y, M, m, b):
        return 2 * (y - M(x, m, b)) * (-dM.db(x, m, b))
    
def cost(X, Y, M, m, b):
    return (1/k) * sum( loss(X[i], Y[i], M, m, b) for i in range(0, k) )

class dcost:
    def dm(X, Y, M, m, b):
        return (1/k) * sum( dloss.dm(X[i], Y[i], M, m, b) for i in range(0, k) )
    def db(X, Y, M, m, b):
        return (1/k) * sum( dloss.db(X[i], Y[i], M, m, b) for i in range(0, k) )

def gd_step(X, Y, M, m, b):
    global l

    cost_pre = cost(X, Y, M, m, b)    
    (m1, b1) = (m - l*dcost.dm(X, Y, M, m, b), b - l*dcost.db(X, Y, M, m, b))
    cost_post = cost(X, Y, M, m1, b1)

    if cost_post > cost_pre:
        print(f"adjusting learning rate from {l} to {l/10}")
        l = l/2
        return m, b
    else:
        return m1, b1

def gd(X, Y, m, b):
    for i in range(0, iterations):        
        m, b = gd_step(X, Y, M, m, b)
    return m, b

if __name__ == "__main__":
    # X = [round(random.uniform(0.0, 10.0),2) for _ in range(k)]    
    # Y = [round(random.uniform(0.0, 10.0),2) for _ in range(k)]
    X = [0.5, 2.3, 2.9]
    Y = [1.4, 1.9, 3.2]    
    m, b = 0, 0

    print("")
    print("")
    print(f"start:\n\tX = {X}\n\tY = {Y}\n\t(m, b) = ({m}, {b})")

    m, b = gd(X, Y, m, b)

    # print(f"(m, b) = ({m}, {b})")
    # print("")

    m_dif = round(abs(m - 0.641026),6)
    b_dif = round(abs(b - 0.948718),6)
    print(f"(m, b) = ({m}, {b}) vs (0.641026, 0.948718) = ({m_dif}, {b_dif})")
    print("")