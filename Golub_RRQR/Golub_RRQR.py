#!/bin/python
# -*- coding:utf-8 -*-
import os;
import sys;

#When as standalone program 
util_dir1  = os.path.split(os.path.realpath(__file__))[0]+"/../utils/Python_Utils";
sys.path.append(util_dir1);
#When as external module
util_dir2  = os.path.split(os.path.realpath(__file__))[0]+"/../../Python_Utils";
sys.path.append(util_dir1);

import Float_Utils;
import Test_Utils;
import Matrix_Utils;
import numpy as np;
import HouseHolder;


#Efficient Algorithm for Computing A Strong Rank-Relealing QR Factorization  850 page 2.RRQR Algorithm   

def check(M,R,PI):
    is_pass = True;
    print "M:";
    Matrix_Utils.matrix_show(M);
    print "R:";
    Matrix_Utils.matrix_show(R);
    print "PI:";
    Matrix_Utils.matrix_show(PI);


    m,n = M.shape;
    MPI = np.zeros([m,n]);
    for j in xrange(n):
        for i in xrange(m):
            MPI[i,j] = M[i,PI[j]];
    print "MPI"
    Matrix_Utils.matrix_show(MPI); 
    
    u,d1,vt = np.linalg.svd(R);
    d       = np.zeros([m,n]);   
    for i in xrange(min(m,n)):
        if not Float_Utils.eq(d1[i],0.0):
            d[i,i] = 1.0/d1[i];

    PINV = np.dot(np.dot(np.transpose(vt),np.transpose(d)),np.transpose(u));
    print "Pinv(R):";
    Matrix_Utils.matrix_show(PINV);
    Q = np.dot(MPI, PINV);

    print "Q=M * PI * pinv(R):";
    Matrix_Utils.matrix_show(Q);

    QtQ = np.dot(np.transpose(Q),Q);
    print "QtQ:(QtQ shall equals I)";
    Matrix_Utils.matrix_show(QtQ);

    Eye = np.zeros([m,m]);
    for i in xrange(min(m,n)):
        Eye[i,i] = 1.0
    if not Matrix_Utils.is_matrix_equals(QtQ,Eye): 
        is_pass = False;

    if True == is_pass:
        print "check ends. Correct!";    
    else:
        print "check ends. Wrong!";

    return is_pass   
 
def test():
    M = np.array([(1,2,3),(2,2,4),(4,5,4)]);
    m = np.copy(M);
    [r,pi,k] = rrqr_float(M,0.1);
    if not check(m,r,pi):
        raise Exception("Check not Pass");

    Test_Utils.test_start_show();
    testM = np.array([[1,0,1,1,0],[0,1,1,1,0],[1,1,1,0,0],[1,0,1,0,0],[0,0,0,0,1]]);
    m     = np.copy(testM);
    [r,pi,k] = rrqr_float(testM,0.001);
    if not check(m,r,pi):
        raise Exception("check not pass");
    print "k=%d"%k;
    Test_Utils.test_end_show();

    Test_Utils.test_start_show();
    [r,pi,k] = rrqr(testM,2);
    print "k=%d"%k;
    if not check(testM,r,pi):
        raise Exception("check not pass");
    Test_Utils.test_end_show();

    Test_Utils.test_start_show();
    testM = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[0,2,3,4],[0,0,3,4],[0,0,0,4]]);
    m     = np.copy(testM);
    [r,pi,k] = rrqr(testM,4);
    print "k=%d"%k;
    if not check(m,r,pi):
        raise Exception("check not pass");
    Test_Utils.test_end_show();    

    Test_Utils.test_start_show();
    testM = np.transpose(m);
    m     = np.transpose(m);
    [r,pi,k] = rrqr(testM,4);
    print "k=%d"%k;
    if not check(m,r,pi):
        raise Exception("check not pass");
    Test_Utils.test_end_show();


def rrqr(R, k):
    (r,c) = R.shape;
    if k > min(r,c):
        raise Exception("k=%d is larger than min(r=%d,c=%d)"%(k,r,c));

    PI   = np.array(range(c));
    for i in xrange(c):
        PI[i] = int(PI[i]);
    S    = [0 for col in xrange(c)];

    for col in xrange(c):
        for row in xrange(r):
            S[col] += R[row,col] * R[row,col];

    for col in xrange(k):
        j,value = maxOf(S,col)
        if Float_Utils.eq(value,0.0):
            return R, PI, col;
        
        tmp      = np.copy(PI[col]);
        PI[col]  = PI[j];
        PI[j]    = tmp;

        tmp    = S[j];
        S[j]   = S[col];
        S[col] = tmp;

        R[:,[j,col]] = R[:,[col,j]];

        R = HouseHolder.HouseHolder_step(R, col);        
        for j in xrange(col,len(S)):
            S[j] -= R[col,j] * R[col,j];

    return R,PI,k

def rrqr_float(R,sigma):
    print R;
    (r,c) = R.shape;
    PI    = np.array(range(c));
    for i in xrange(c):
        PI[i] = int(PI[i]);
        
    S    = [0 for col in xrange(c)];
    for col in xrange(c):
        for row in xrange(r):
            S[col] += R[row,col] * R[row,col];

    for k in xrange(min(r,c)):
        j,value = maxOf(S,k)
        if Float_Utils.lt(value, sigma):   
            return R, PI, k;

        #swap j and col
        tmp     = np.copy(PI[j]);
        PI[j]   = PI[k];
        PI[k] = tmp;

        tmp  = np.copy(S[j]);
        S[j] = S[k];
        S[k] = tmp;

        R[:,[j,k]] = R[:,[k,j]];

        R = HouseHolder.HouseHolder_step(R, k);        
        print "after k=%d,R"%k;
        Matrix_Utils.matrix_show(R);
        print "PI:";
        Matrix_Utils.matrix_show(PI);
        for i in xrange(k,len(S)):
            S[i] -= R[k,i] * R[k,i];

    return R,PI,min(r,c);

def maxOf(S,k):
    max_p = k;
    max_v = S[k];
    for i in xrange(k+1,len(S)):
        if S[i] > max_v:
            max_p = i;
            max_v = S[i];
    return max_p,max_v;

if __name__ == "__main__":
    test();
