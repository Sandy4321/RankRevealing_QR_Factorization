#/bin/python
import os;
import sys;
golub_dir  = os.path.split(os.path.realpath(__file__))[0]+"/../Golub_RRQR";
sys.path.append(golub_dir);
#When as standalone program 
util_dir1  = os.path.split(os.path.realpath(__file__))[0]+"/../utils/Python_Utils";
sys.path.append(util_dir1);
#When as external module
util_dir2  = os.path.split(os.path.realpath(__file__))[0]+"/../../Python_Utils";
sys.path.append(util_dir1);
from numpy import *;
from math  import *;
from Matrix_Utils import *;
from Float_Utils import *;
from Test_Utils import *;
import HouseHolder;
import Golub_RRQR;
import numpy as np;

np.random.seed(0);

is_debug=False;


def check_final(R,k,f=1.414):
    r,c = R.shape;
    Ak = R[0:k,0:k];
    Bk = R[0:k,k:c];
    Ck = R[k:r,k:c];
    pkn = math.sqrt(1+ f*f*c*(c-k))
    isPass = True;

    print "final check starts";
    u,sigma_R,v  = linalg.svd(R);
    u,sigma_Ak,v = linalg.svd(Ak);
    if k != c:
        u,sigma_Ck,v = linalg.svd(Ck);
    else:
        sigma_Ck = array([]);

    for i in xrange(k):
        if gt(sigma_Ak[i], sigma_R[i]):
            print "sigma_Ak[%d] > sigma_R[%d]"%(i,i);
            isPass = False;
        if lt(sigma_Ak[i], sigma_R[i] / pkn):
            print "sigma_Ak[%d] < sigma_R[%d]/pkn"%(i,i);
            isPass = False;

    for j in xrange(c-k):
        if lt(sigma_Ck[j], sigma_R[k+j]):
            print "sigma_Ck[%d] < sigma_R[%d]"%(j,k+j);
            isPass = False;
        if gt(sigma_Ck[j], pkn * sigma_R[k+j]):
            print "sigma_Ck[%d] > pkn*sigma_R[%d]"%(j,k+j);
            isPass = False;

    if False == isPass: 
        print "k:",k;
        print "pkn:",pkn;
        print "sigma_R:";
        matrix_show(sigma_R);
        print "sigma_Ak:";
        matrix_show(sigma_Ak);
        print "sigma_Ck:";
        matrix_show(sigma_Ck);
    

    print "final check ends\n";

    return isPass;

def check_step(R, invA_B, omega, gamma, k):
    isPass = True;
    r,c = R.shape;
    Ak = R[0:k,0:k];
    Bk = R[0:k,k:c];
    Ck = R[k:r,k:c];
    
    print "check start: k = ",k;

    invAk = linalg.inv(Ak);
    check_omega = array([0.0 for i in xrange(r)]);
    for i in xrange(k):
        for j in xrange(k):
            check_omega[i] += invAk[i][j] * invAk[i][j];
        check_omega[i] = math.sqrt(check_omega[i]);
    if not is_matrix_equals(check_omega, omega):
        print "omega:";
        matrix_show(omega);
        print "check_omega:";
        matrix_show(check_omega);
        isPass  = False;

    a = check_gamma = array([0.0 for i in xrange(c)]);
    for j in xrange(k,c):
        for i in xrange(k,r):
            check_gamma[j] += R[i,j] * R[i,j];
        check_gamma[j] = math.sqrt(check_gamma[j]);
    if not is_matrix_equals(check_gamma, gamma):
        print "gamma:";
        matrix_show(gamma);
        print "check_gamma:";
        matrix_show(check_gamma);
        isPass = False;

    check_invA_B =  dot(linalg.inv(Ak),Bk);    
    if not is_matrix_equals(invA_B, check_invA_B):
        print "invA_B:";
        matrix_show(invA_B);
        print "check_invA_B:";
        matrix_show(check_invA_B);
        isPass = False;

    print "end check";
    print "";
    return isPass;

def show_step(R, invA_B, omega, gamma, k):
    print "R:"
    matrix_show(R);
    print "gamma:"
    matrix_show(gamma);
    print "omega:"
    matrix_show(omega); 
    print "";



def rrqr(R, k, f=1.414):
    m,n = R.shape;
    if m < n:
        raise Exception("Strong Rank Revealing QR Factorization requires m >= n, but m=%d, n=%d"%(m,n));
    if k > n or k <= 0:
        raise Exception("Strong Rank Revealing QR Factorization requires 1<= k <= n, but n=%d, k=%d"%(n,k));

    if True == is_debug:
        print "start:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
        print "R:";
        matrix_show(R);
        print "k:",k;
        print "";
        #check_final(R, k, f);
    
    PI = array([]);
    if True == is_debug:
        M  = copy(R);
        PI = array(range(n));
        for i in xrange(k+1):
            R = HouseHolder.HouseHolder_step(R,i);
    else:     
        R,PI,k1 = Golub_RRQR.rrqr(R, k);
        k = k1;
    

    Ak   = R[0:k,  0:k];
    Bk   = R[0:k,  k:n]; 
    invA = linalg.inv(Ak);
    omega = array([0.0 for i in xrange(m)]);
    for i in xrange(k):
        for j in xrange(k):
            omega[i] += invA[i][j] * invA[i][j];
        omega[i] = math.sqrt(omega[i]);
        
    gamma = array([0.0 for j in xrange(n)]);
    for j in xrange(k,n):
        for i in xrange(k,m):
            gamma[j] += R[i][j] * R[i][j];
        gamma[j] = math.sqrt(gamma[j]);

    invA_B = dot(invA, Bk);

    if True == is_debug:
        print "After HouseHolder";
        show_step(R, invA_B, omega, gamma, k);    
        check_step(R, invA_B, omega, gamma, k);

    flag,i,j = is_rho_less_f(invA_B, omega, gamma, k, f);
    while not flag:
        R, PI, invA_B, omega, gamma =  \
        update_swap_k_kplusj(R, PI, invA_B, omega, gamma, k, j);
        if True == is_debug:
            print "swap_k_kplusj: j = ",j,"  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
            show_step(R, invA_B, omega, gamma, k);
            if not check_step(R, invA_B, omega, gamma, k):
                print "Test fails. Please debug this matrix:";
                matrix_show(M);
                exit(0);

        R, PI, invA_B, omega, gamma =  \
        update_shift(R, PI, invA_B, omega, gamma, i, k);
        if True == is_debug:
            print "shift:i = ",i, "   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            show_step(R, invA_B, omega, gamma,k);
            if not check_step(R, invA_B, omega, gamma, k):
                print "Test fails. Please debug this matrix:";
                matrix_show(M);
                exit(0);                

        R, PI, invA_B, omega, gamma =  \
        update_swap_kminus1_k(R, PI, invA_B, omega, gamma,k);           
        if True == is_debug:
            print "swap_kminus1_k  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            show_step(R, invA_B, omega, gamma, k);
            if not check_step(R, invA_B, omega, gamma, k):
                print "Test fails. Please debug this matrix:";
                matrix_show(M);
                exit(0);

        flag, i, j = is_rho_less_f(invA_B, omega, gamma, k ,f); 
    
    if True == is_debug:
        if not check_final(R, k, f):
            print "Test fails. Please debug this matrix:";
            matrix_show(M);
            exit(0);
    return R, PI, k;


def is_rho_less_f(invA_B, omega, gamma, k, f=1.414):    
    r,c = invA_B.shape;
    for i in xrange(r):
        for j in xrange(c):
            rho = invA_B[i][j] * invA_B[i][j] + \
                  gamma[j+k]*gamma[j+k] * omega[i] * omega[i];
            if rho > f:
                return False, i, j;

    return True,None,None;
        


def update_swap_k_kplusj(R, PI, invA_B, omega, gamma, k, j): 

    #swap column k and column k + j;  
    row,col = R.shape; 
    if j == 0 or k+j >= col:
        return R, PI, invA_B, omega, gamma,;
    print PI.shape;
    print k;
    tmp     = copy(PI[k]);
    PI[k]   = PI[k+j];
    PI[k+j] = tmp;

    tmp             = copy(R[:,k:k+1]);
    R[:,k:k+1]      = R[:,k+j:k+j+1];
    R[:,k+j:k+j+1]  = tmp;    
    ##employ Given Rotation to ensure Ck[:,0] = [notzero, 0,...0]
    ##Ck[:,0] = [notzero, 0, ..., 0] is important to Update_swap_kminus1_k
    for i in xrange(1,row - k):
        lou = math.sqrt(R[k,k]*R[k,k] + R[k+i,k]*R[k+i,k]);
        c   = R[k,k] / lou;
        s   = -1 * R[k+i,k] / lou;
        for p in xrange(k,col):
            up   = c*R[k,p] - s*R[k+i,p];
            down = s*R[k,p] + c*R[k+i,p];
            R[k,p]   = up;
            R[k+i,p] = down;
        R[k,k] = lou;
       
 
    tmp         = gamma[ k ];
    gamma[k]    = gamma[ k+j ];
    gamma[k+j]  = tmp;

    tmp             =  copy(invA_B[:,j:j+1]);
    invA_B[:,j:j+1] =  invA_B[:,0:1];
    invA_B[:,0:1]   =  tmp;
    
    return R, PI, invA_B , omega, gamma;

def update_shift(R, PI, invA_B, omega, gamma, i, k):
    #4.2
    m,n = R.shape;
    if i >= k:   return R, PI, invA_B, omega, gamma

    tmp = copy(PI[i]);
    for idx in xrange(i,k-1):
        PI[idx] = PI[idx+1];
    PI[k-1] = tmp;
    
    tmp = copy(R[0:k,i:i+1]);
    for idx in xrange(i,k-1):
        R[0:k, idx:idx+1] = R[0:k, idx+1:idx+2];
    R[0:k,k-1:k] = tmp;

    
    ## Given Rotation
    for idx in xrange(i,k-1):
        r = math.sqrt(R[idx][idx] * R[idx][idx] + R[idx+1][idx]*R[idx+1][idx]);
        c = R[idx, idx] / r;
        s = -R[idx+1, idx] / r;
        for j in xrange(idx,n):
            up   = c * R[idx][j] - s * R[idx+1][j];
            down = s * R[idx][j] + c * R[idx+1][j];
            R[idx, j]   = up;
            R[idx+1, j] = down;      

    tmp = omega[i];
    for idx in xrange(i,k-1):
        omega[idx] = omega[idx+1];
    omega[k-1] = tmp;

    tmp = copy(invA_B[i:i+1,:]);
    for idx in xrange(i,k-1):
        invA_B[idx:idx+1,:] = invA_B[idx+1:idx+2,:];
    invA_B[k-1:k,:] = tmp;
    
    return R, PI, invA_B, omega, gamma;

 
def update_swap_kminus1_k(R,PI,invA_B, omega,gamma,k):
    tmp     = copy(PI[k-1]);
    PI[k-1] = PI[k];
    PI[k]   = tmp;

    row,col = R.shape;

    if 1 == k:
        tmp        = copy(R[0:2,0:1]);
        R[0:2,0:1] = R[0:2, 1:2];
        R[0:2,1:2] = tmp;
        
        ##Given Rotation
        lou = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0]);
        c   = R[0,0] / lou;
        s   = -R[1,0] / lou;
        for i in xrange(col):
            up   = c * R[0,i] - s * R[1,i];
            down = s * R[0,i] + c * R[1,i];
            if 0 != i and 1 != i:
                gamma[i] = math.sqrt(gamma[i] * gamma[i] - R[1,i] * R[1,i] + down * down);
            R[0,i] = up;
            R[1,i] = down;
        gamma[1] = abs(R[1,1]);

        omega[0] = abs(1.0/R[0,0]); 
        
        for i in xrange(1,col):
            invA_B[0,i-1] = R[0,i] / R[0,0];
        return R, PI, invA_B, omega, gamma; 
        


    r  = R[k-1][k-1];
    mu = R[k-1][k] / r;
    nu = R[k][k]   / r;
    lou = math.sqrt(mu * mu + nu * nu);


    invA_B_r,invA_B_c = invA_B.shape;
    Akminus1 = copy(R[0:k-1, 0:k-1]);
    b1       = copy(R[0:k-1, k-1:k]);
    u        = dot(linalg.inv(Akminus1),b1);
    u1       = invA_B[0:k-1,0:1];
    u2       = transpose(invA_B[k-1:k, 1:invA_B_c]);
    U        = invA_B[0:k-1, 1:invA_B_c];

    #R and gamma
    tmp             = copy(R[0:k-1, k-1:k])
    R[0:k-1, k-1:k] = R[0:k-1, k:k+1]
    R[0:k-1, k:k+1] = tmp; 
    R[k-1][k-1] = r * lou;
    R[k-1][k]   = r * mu / lou;
    R[k][k]     = r * nu / lou;
    gamma[k]    = abs(R[k,k]);

    for i in xrange(k+1,col):
        c1 = R[k-1][i] * mu / lou + R[k][i] * nu / lou;
        c2 = R[k-1][i] * nu / lou - R[k][i] * mu / lou;
        gamma[i]  = math.sqrt(gamma[i] * gamma[i] + \
                              c2 * c2 - R[k][i] * R[k][i]); 
        R[k-1][i] = c1;
        R[k][i]   = c2;

    
    #omega
    hatr = R[k-1][k-1];
    omega[k-1] = abs(1/hatr);
    for i in xrange(k-1):
        omega[i] = math.sqrt(omega[i]*omega[i] \
                   - u[i]*u[i]/r/r + (u1[i]+mu*u[i])*(u1[i]+mu*u[i])/hatr/hatr );

    #invA_B
    c1         = R[k-1:k, k+1:col];
    c2         = R[k:k+1, k+1:col];
    new_invA_B = copy(invA_B);
    invA_B_row, invA_B_col           = new_invA_B.shape; 
    new_invA_B[k-1][0]               = mu/lou/lou;
    new_invA_B[k-1:k, 1:invA_B_col]  = copy(R[k-1:k, k+1:col])/hatr;
    new_invA_B[0:k-1, 0:1         ]  = (nu*nu*u - mu*u1)/lou/lou;
    new_invA_B[0:k-1, 1:invA_B_col]  = U + (nu*dot(u,c2) - dot(u1,c1)) / hatr; 
    return R, PI, new_invA_B, omega, gamma;    

if __name__ == "__main__":
 
    is_debug = True;

    test_start_show();
    print "Kahan example1, n = 10, k = 9, psi = 0,8, ksi=0.6. I check kahan example1 with these settings doesn't satisfy the conditions of RRQR  ";
    sn = zeros([10,10]);
    sn[0,0] = 1;
    for i in xrange(1,10):
        sn[i,i] = 0.6 * sn[i-1,i-1];
    kn = zeros([10,10]);
    for i in xrange(10):
        for j in xrange(10):
            if i == j:  kn[i,j] = 1;
            if i < j :  kn[i,j] = -0.8;
    testM = dot(sn,kn);
    R     = rrqr(testM, 9);
    test_end_show();


    test_start_show();
    testM = array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[0,2,3,4],[0,0,3,4],[0,0,0,4]]);
    R     = rrqr(testM, 4);
    test_end_show();

    #test_start_show();
    #m = matrix_read("./testdata");
    #print m.shape;
    #R,PI = rrqr(m,2);
    #test_end_show();

    test_start_show();
    testM = array([[1,0,1,1],[0,1,1,1],[1,1,1,0],[1,0,1,0]]);
    R = rrqr(testM,3);
    test_end_show();


    test_start_show();
    testM[3,3] = 10;
    R = rrqr(testM,2);
    test_end_show();


    test_start_show();
    testM = array([[1,0,1,1,1],[0,1,1,1,1],[1,1,1,0,1],[1,0,1,0,1],[10,10,10,10,10]]);
    R = rrqr(testM,3);
    test_end_show();

    test_start_show();
    R = rrqr(testM,1);
    test_end_show();

    for i in xrange(100):
        for k in xrange(1,6): 
            test_start_show();
            testM = np.random.rand(6,6);
            R = rrqr(testM,k);
            test_end_show();

    
    for i in xrange(100):
        for k in xrange(1,7):
            test_start_show();
            testM = np.random.rand(7,7);
            R = rrqr(testM,k);
            test_end_show();   
    
    for i in xrange(100):
        for k in xrange(1,8):
            test_start_show();
            testM = np.random.rand(1000,100);
            R = rrqr(testM,k+20);
            test_end_show(); 
