import sys
#sys.path.append('../')
import torch
import torch.nn as nn
from qpth.qp import QPFunction
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle
import networkx
import numpy as np
import scipy



class MatchingLayer(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, nNodes=1, eps=1e-4):
        super(MatchingLayer, self).__init__()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps   = eps
        self.N = nNodes
        N      = nNodes

        # Empty Tensor
        e = Variable(torch.Tensor())

        ROWlhs    = Variable( torch.zeros(N,N**2)  )
        ROWrhs    = Variable(  ( torch.ones(N) )    )
        COLlhs    = Variable( torch.zeros(N,N**2)  )
        COLrhs    = Variable(  ( torch.ones(N) )    )
        # All values are positive
        POSlhs    = Variable(    -torch.eye(N**2,N**2)        )
        POSrhs    = Variable(    -torch.zeros(N**2)        )

        """
        print("Inside constructor")
        print("ROWlhs = ")
        print( ROWlhs )
        print("COLlhs = ")
        print( COLlhs )
        """




        # Row sum constraints
        for row in range(N):
            ROWlhs[row,row*N:(row+1)*N] = 1.0

        # Column sum constraints
        for col in range(N):
            COLlhs[col,col:-1:N] = 1.0
        # fix the stupid issue of bottom left not filling
        COLlhs[-1,-1] = 1.0

        """
        print("After filling:")
        print("ROWlhs = ")
        print( ROWlhs )
        print("COLlhs = ")
        print( COLlhs )
        """


        # Total inequalities
        self.G = torch.cat( (ROWlhs,COLlhs,POSlhs),0  )
        self.h = torch.cat( (ROWrhs,COLrhs,POSrhs),0  )
        self.Q = self.eps*Variable(torch.eye(self.N**2))

        """
        print("After stacking:")
        print("self.G = ")
        print( self.G )
        print("self.h = ")
        print( self.h )
        """


        self.nineq = self.G.size(0)
        #self.neq   = self.A.size(0)


        # decision variable yij    1 <= i <= N1,  1 <= j <= N2
        # N1 nodes in G1, N2 nodes in G2
        #
        # max v^T y
        #
        # sum_i yij <= 1    all j in G1 vertices
        # sum_j yij <= 1    all i in G2 vertices
        #
        # yij >= 0  (is this necessary?)
        #
        #
        # ineq matrix w/ N rows, N^2 cols
        #
        # assume the variable y is an unrolled version of the matrix above
        # row-major order (row1, row2, ...)
        #
        # v[i*J+j] = v_ij is the weight matching node i on the left to j on the right





    def forward(self, x):
        nBatch = x.size(0)

        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())



        # Try these with and without the expand
        Q = self.Q.to(self._device)  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        G = self.G.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        h = self.h.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )


        A = e.to(self._device)
        b = e.to(self._device)

        """
        print("Inside matching layer forward:")
        print("Q = ")
        print( Q )
        print("G = ")
        print( G )
        print("h = ")
        print( h )
        print("A = ")
        print( A )
        print("b = ")
        print( b )
        print("x = ")
        print( x )
        """

        """
        print("Inside matching layer forward:")
        print("Q.size() = ")
        print( Q.size() )
        print("G.size() = ")
        print( G.size() )
        print("h.size() = ")
        print( h.size() )
        print("A.size() = ")
        print( A.size() )
        print("b.size() = ")
        print( b.size() )
        print("x.size() = ")
        print( x.size() )
        """

        """
        print("Q.device = ")
        print( Q.device )
        print("G.device = ")
        print( G.device )
        print("h.device = ")
        print( h.device )
        print("A.device = ")
        print( A.device )
        print("b.device = ")
        print( b.device )
        print("x.device = ")
        print( x.device )
        print("self._device= ")
        print( self._device )
        """



        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A, b   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        return x.float()






# Takes adjacency matrices G1 and G2 (np.ndarray's squeezed to min dimension, no batch dimension (2D) )
def compute_alignment(G1,G2,s1=1,s2=0,s3=0):

    ONE = np.ones(G1.shape)
    A = (s1 + s2 - 2*s3)*np.kron(G1,G2) + (s3-s2)*np.kron(G1,ONE) + (s3-s2)*np.kron(ONE,G2) + s2*np.kron(ONE,ONE)

    return A


# takes an alignment matrix (np.ndarray dim 2), returns matching permutation
def EigenAlign(A,matching_layer):

    # compute top eigenvector of A
    # use it as input to MatchingLayer, eps = 0
    vals, vecs = np.linalg.eig(A)
    v = np.real(vecs[0])    # top eigenvector

    v = torch.Tensor(v).unsqueeze(0) #create a tensor batch

    P = matching_layer(v)


    return P








if __name__ == "__main__":


    input = pickle.load(open('pickle_input.p','rb'))

    """
    print("input = ")
    print(input)
    print("len(input) = ")
    print( len(input) )
    print("input[0] = ")
    print( input[0] )
    print("input[1] = ")
    print( input[1] )
    print("input[0][0] = ")
    print( input[0][0] )
    print("input[0][1] = ")
    print( input[0][1] )

    W0 = input[0][0]
    W1 = input[1][0]
    mat0 = W0[0][1]
    mat1 = W1[0][1]
    print("mat0 = ")
    print( mat0 )
    print("mat1 = ")
    print( mat1 )
    """




    #noise = np.random.uniform(0.000, 0.050, 1)
    #noise1 = self.ErdosRenyi_netx(pe1, self.N)
    #W_noise = W*(1-noise1) + (1-W)*noise2



    #pe1 = self.noise
    #pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
    #noise1 = self.ErdosRenyi_netx(pe1, self.N)
    #noise2 = self.ErdosRenyi_netx(pe2, self.N)
    #W_noise = W*(1-noise1) + (1-W)*noise2





    N = 10
    edge_density = 0.2
    noise = 0.07
    pe1 = noise
    pe2 = (edge_density*noise)/(1.0-edge_density)


    g = networkx.erdos_renyi_graph(N, edge_density)
    W = networkx.adjacency_matrix(g).todense().astype(float)
    W = np.array(W)

    g_noise1 = networkx.erdos_renyi_graph(N, pe1)
    g_noise2 = networkx.erdos_renyi_graph(N, pe2)

    noise1 = networkx.adjacency_matrix(g_noise1).todense().astype(float)
    noise2 = networkx.adjacency_matrix(g_noise2).todense().astype(float)
    noise1 = np.array(noise1)
    noise2 = np.array(noise2)
    W_noise = W*(1-noise1) + (1-W)*noise2



    print("W = ")
    print( W   )
    print("W_noise = ")
    print( W_noise   )

    s1 = 1
    s2 = s3 = 0
    G1 = W
    G2 = W_noise


    ONE = np.ones(G1.shape)


    print("G1.shape = ")
    print( G1.shape )
    print("ONE.shape = ")
    print( ONE.shape )

    A = (s1 + s2 - 2*s3)*np.kron(G1,G2) + (s3-s2)*np.kron(G1,ONE) + (s3-s2)*np.kron(ONE,G2) + s2*np.kron(ONE,ONE)
    #A = np.kron(G1,G2)
    print("A = ")
    print( A )

    matching_layer = MatchingLayer(nNodes=N, eps=1e-6)

    P = EigenAlign(A,matching_layer)
    print("P = ")
    print( P )



    """
    vals, vecs = np.linalg.eig(A)
    print("vecs = ")
    print( np.real(vecs) )
    print("vals = ")
    print( np.real(vals) )

    print("vecs[0] = ")
    print( np.real(vecs[0]) )
    """

    """
    print("W = ")
    print( W   )
    print("noise1 = ")
    print( noise1    )
    print("noise2 = ")
    print( noise2    )
    print("W_noise = ")
    print( W_noise   )
    """























    """
    print("noise1 = ")
    print( noise1 )
    print("noise2 = ")
    print( noise2 )
    print("(1-noise1) = ")
    print( (1-noise1)    )
    print("W = ")
    print( W )
    #print("(1-W) = ")
    #print( (1-W) )
    AA = (1-noise1)
    BB = W
    print("W*(1-noise1) = ")
    print( W*(1-noise1)    )
    #print("(1-W)*noise2 = ")
    #print( (1-W)*noise2    )
    #print("(1-W)*noise2 = ")
    #print( (1-W)*noise2    )
    print("W_noise = ")
    print( W_noise )
    print("type(W) = ")
    print( type(W) )
    A = np.ones((10,10))
    B = np.ones((10,10))
    print("A*B = ")
    print( A*B    )
    print("type(A*B) = ")
    print( type(A*B)    )
    print("AA = ")
    print( AA    )
    print("BB = ")
    print( BB    )
    print("AA*BB = ")
    print( AA*BB    )
    print("np.array(AA)*np.array(BB) = ")
    print( np.array(AA)*np.array(BB) )
    print("type(AA) = ")
    print( type(AA)    )
    print("type(BB) = ")
    print( type(BB)    )
    """

























    """
    n = 3

    vlist = []

    v = torch.eye(n,n)
    #v = v.flatten()
    #v = v.unsqueeze(0)    # make v a batch

    print("v = ", v)
    vlist.append(v.flatten()) #.unsqueeze(0))

    v = v[[0,2,1]]

    print("v = ", v)
    vlist.append(v.flatten()) #.unsqueeze(0))

    v = torch.stack(vlist)

    matching_layer = MatchingLayer(nNodes=n, eps=1e-4)

    match = matching_layer(v)
    print("match = ", match)
    """
