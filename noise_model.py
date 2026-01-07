#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma, polygamma
#import statsmodels.api as sm

class Noise_Model:
    
    def __init__(self, response, covariates, constant, distribution):
        self.response = response
        self.X = covariates #X
        self.C = constant #size ncells
        self.distribution = distribution
        self.ncells, self.ncovs = np.shape(self.X)    
    
    def loglikelihood(self, params):
        
        if self.distribution == "Poisson":
            lam = params
            logl = self.response * np.log(lam)-lam-gammaln(self.response+1)
            return logl
        
        elif self.distribution == "NB" or self.distribution == "GeneralDispersion":
            mu = params[:,0]
            alpha = params[:,1]
            logl = self.response * np.log(alpha * mu)
            logl -= (self.response + 1/alpha) * np.log(1 + alpha * mu)
            logl += gammaln(self.response + 1/alpha)
            logl -= gammaln(1 / alpha)
            logl -= gammaln(self.response + 1)
            return logl
        
        elif self.distribution == "ZeroPoisson":
            p = params[:, 0]
            lam = params[:, 1]
            logl = np.zeros(self.ncells)
            indices_zero = np.where(self.response < 0.1)
            indices_pos = np.where(self.response > 0.1)
            p_zero = p[indices_zero]
            logl[indices_zero] = np.log(p_zero + (1-p_zero)*np.exp(-lam[indices_zero]))
            logl[indices_pos] = np.log(1-p[indices_pos]) + self.response[indices_pos] * np.log(lam[indices_pos])-lam[indices_pos]-gammaln(self.response[indices_pos]+1)
            return logl
                
        else:
            # If none of the previous is chosen, use Normal with log link
            mu = params[:, 0]
            sigma = params[:, 1]
            logl = -np.log(2*np.pi*sigma)
            logl -= (self.response-mu)**2 / sigma
            logl /= 2
            return logl

    
    def score_statistic(self, params):
        
        if self.distribution == "Poisson":
            lam = params
            return self.response-lam
        
        elif self.distribution == "NB":
            mu = params[:,0]
            alpha = params[:,1]
            u_mu = (self.response - mu) * mu / (mu + alpha * mu**2)
            u_alpha = -np.sum(digamma(self.response + 1/alpha) - digamma(1/alpha) - np.log(1 + alpha * mu)  - \
                     - alpha * (self.response - mu)/(1 + alpha*mu)) / (alpha[0]**2)                 
            return u_mu, u_alpha
        
        elif self.distribution == "ZeroPoisson":
            p = params[:, 0]
            lam = params[:, 1]
            eps = 0.01
            indices_zero = np.where(self.response < eps)
            indices_pos = np.where(self.response > eps)
            u_p = np.zeros(self.ncells)
            u_p[indices_zero] = (1-p[indices_zero]) * p[indices_zero] * (1-np.exp(-lam[indices_zero])) / (p[indices_zero] + (1-p[indices_zero])*np.exp(-lam[indices_zero]))
            u_p[indices_pos] = -p[indices_pos] 
            u_l = np.zeros(self.ncells)
            u_l[indices_zero] = - np.exp(-lam[indices_zero]) * lam[indices_zero] * (1-p[indices_zero]) / (p[indices_zero] + (1-p[indices_zero])*np.exp(-lam[indices_zero]))
            u_l[indices_pos] = self.response[indices_pos] - lam[indices_pos]
            return u_p, u_l
        
        elif self.distribution == "GeneralDispersion":
            mu = params[:,0]
            alpha = params[:,1]
            u_mu = (self.response - mu) * mu / (mu + alpha * mu**2) 
            return u_mu
        
        else:
            mu = params[:, 0]
            sigma = params[:, 1]
            return mu * (self.response - mu) / sigma
            
    def hessian_weights(self, params):
        
        if self.distribution == "Poisson":
            return params
        
        elif self.distribution == "NB":
            mu = params[:,0]
            alpha = params[:,1]
            w_mu = mu**2 / (mu + alpha * mu**2)
            u_alpha =  -(digamma(self.response + 1/alpha) - digamma(1/alpha) - np.log(1 + alpha * mu)  - \
                     - alpha * (self.response - mu)/(1 + alpha*mu)) / (alpha**2)
            w_alpha = np.sum(-(2/alpha) *  u_alpha + 1/alpha**4 * (polygamma(1, self.response + 1/alpha) - \
                    polygamma(1, 1/alpha) - alpha - 1/(1/alpha + mu) + \
                    (self.response - mu)/(mu + 1/alpha)**2))
            return w_mu, w_alpha
        
        elif self.distribution == "ZeroPoisson":
            p = params[:, 0]
            lam = params[:, 1]
            eps = 0.01
            indices_zero = np.where(self.response < eps)
            indices_pos = np.where(self.response > eps)
            w_p = np.zeros(self.ncells)
            p_zero = p[indices_zero]
            p_pos = p[indices_pos]
            num_p = (1-np.exp(-lam[indices_zero]))
            den_p = p_zero + (1-p_zero)*np.exp(-lam[indices_zero])
            w_p[indices_zero] = ((num_p/den_p) * p_zero * (1-p_zero))**2 - (1-p_zero) * (1-2*p_zero) * p_zero * num_p / den_p  
            w_p[indices_pos] = p[indices_pos] * (1-p[indices_pos])
            w_l = np.zeros(self.ncells)
            num_l = (1-p_zero) * np.exp(-lam[indices_zero])
            den_l = p_zero + (1-p_zero) * np.exp(-lam[indices_zero])
            w_l[indices_zero] = (num_l/den_l) * lam[indices_zero] * (1 + (num_l/den_l) * lam[indices_zero])
            w_l[indices_pos] = lam[indices_pos]
            return w_p, w_l
        
        elif self.distribution == "GeneralDispersion":
            mu = params[:,0]
            alpha = params[:,1]
            w_mu = mu**2 / (mu + alpha * mu**2)
            return w_mu
        
        else:
            mu = params[:, 0]
            sigma = params[:, 1]
            return mu**2 / sigma
    
    
    def iterate(self, coef):
        
        if self.distribution == "Poisson":
            tmp=np.matmul(self.X, coef)
            params = np.exp(tmp+self.C)

        
        elif self.distribution == "NB":
            mu = np.exp(np.matmul(self.X, coef[:-1])+self.C)
            alpha = coef[-1]
            params = np.concatenate((mu.reshape(-1,1), alpha*np.ones((self.ncells,1))), axis = 1)
            u_mu, u_alpha = self.score_statistic(params)
            w_mu, w_alpha = self.hessian_weights(params)           
            z_mu = np.matmul(self.X, coef[:-1]) + u_mu/w_mu
            B_mu = np.linalg.inv(np.matmul(self.X.T*w_mu,self.X))
            v_mu = np.matmul(self.X.T * w_mu, z_mu)
            coef_mu = np.matmul(B_mu, v_mu)
            coef_alpha = alpha - u_alpha / w_alpha
            return np.concatenate((coef_mu, [coef_alpha]))

        elif self.distribution == "ZeroPoisson":
            p = np.exp(np.matmul(self.X, coef[:,0])) / (1 + np.exp(np.matmul(self.X, coef[:,0])))
            p = p.reshape((self.ncells, 1))
            lam = np.exp(np.matmul(self.X, coef[:,1])).reshape((self.ncells, 1))
            params = np.concatenate((p, lam), axis = 1)
            u_p, u_l = self.score_statistic(params)
            w_p, w_l = self.hessian_weights(params)
            
            z_p = np.matmul(self.X, coef[:,0]) + u_p/w_p
            B_p = np.linalg.inv(np.matmul(self.X.T*w_p,self.X))
            v_p = np.matmul(self.X.T * w_p, z_p)
            coef_p = np.matmul(B_p, v_p).reshape((-1, 1))
            
            z_l = np.matmul(self.X, coef[:,1]) + u_l/w_l
            B_l = np.linalg.inv(np.matmul(self.X.T*w_l,self.X))
            v_l = np.matmul(self.X.T * w_l, z_l)
            coef_l = np.matmul(B_l, v_l).reshape((-1, 1))
            
            return np.concatenate((coef_p, coef_l), axis = 1)
        
        elif self.distribution == "GeneralDispersion":
            mu = np.exp(np.matmul(X, coef))
            h = 0.1
            l = 2*np.pi/h
            counts = np.zeros(int(l)+1)
            counts_est = np.zeros(int(l)+1)
            var_est = np.zeros(int(l)+1)

            for i in range(int(l)+1):
                sub_col = self.response[np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()]
                sub_est = mu[np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()]
                counts[i] = np.mean(sub_col)
                counts_est[i] = np.mean(sub_est)
                var_est[i] = np.mean((sub_col-sub_est)**2)
            
            a, b, c = np.polyfit(counts_est, var_est, 2)
            alpha = np.zeros(self.ncells)
            for i in range(int(l)+1):
                indices = np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()
                alpha[indices] = (b-1)/counts_est[i] + a + c/counts_est[i]**2
        
            params = np.concatenate((mu.reshape(-1,1), alpha.reshape((-1, 1))), axis = 1)
            u_mu = self.score_statistic(params)
            w_mu = self.hessian_weights(params)           
            z_mu = np.matmul(self.X, coef) + u_mu/w_mu
            B_mu = np.linalg.inv(np.matmul(self.X.T*w_mu,self.X))
            v_mu = np.matmul(self.X.T * w_mu, z_mu)
            coef_mu = np.matmul(B_mu, v_mu).reshape(-1)   
            return coef_mu
        
        else:
            mu = np.exp(np.matmul(X , coef))
            sigma = np.sum((self.response - mu)**2) / self.ncells
            params = np.concatenate((mu.reshape((-1,1)), sigma*np.ones((self.ncells, 1))), axis = 1)
            
        # THIS WAS ADDED BY ME REMOVE THE IF AND UNCOMMENT THE Z IF PROBLEM 
        
        hessian = self.hessian_weights(params)
        
        if np.any(hessian < 1e-8):
            print("Warning: Hessian values are too small, adjusting to avoid division by zero")
            hessian[hessian < 1e-8] = 1e-8
        # 
        #z = np.matmul(self.X, coef) + self.score_statistic(params)/self.hessian_weights(params)
        z = np.matmul(self.X, coef) + self.score_statistic(params)/hessian

        try:
            B = np.linalg.inv(np.matmul(self.X.T*self.hessian_weights(params), self.X))
            v = np.matmul(self.X.T * self.hessian_weights(params), z)
            
            new_coef = np.matmul(B, v)
            
            return new_coef.reshape(-1)
        except:
            print('except')
            return coef
    
    def fit(self, n_iter, alpha_init=1.0):
        
        if self.distribution == "ZeroPoisson":
            coef = np.zeros((self.ncovs, 2))
            coef[0, 0] = 1
            coef[0, 1] = 1
        
        elif self.distribution == "NB":
            coef = np.zeros(self.ncovs+1)
            coef[0] = np.log(np.mean(self.response))
            coef[-1] = alpha_init #initilize alpha
            
        else:
            coef = np.zeros(self.ncovs)
            coef[0] = np.log(np.mean(self.response))
        
    
        for i in range(n_iter):
            coef = self.iterate(coef)
            
        
        return coef 
    
    def regularized_iteration(self, coef, tau):
        
        if self.distribution == "Poisson":
            params = np.exp(np.matmul(self.X, coef)+self.C)
        
        elif self.distribution == "NB":
            mu = np.exp(np.matmul(self.X, coef[:-1])+self.C)
            alpha = coef[-1]
            params = np.concatenate((mu.reshape(-1,1), alpha*np.ones((self.ncells,1))), axis = 1)
            u_mu, u_alpha = self.score_statistic(params)
            w_mu, w_alpha = self.hessian_weights(params)           
            z_mu = np.matmul(self.X, coef[:-1]) + u_mu/w_mu
            B_mu = np.matmul(self.X.T*w_mu,self.X)+tau*np.identity(self.ncovs)
            v = np.matmul(self.X.T * w_mu, z_mu)
            coef_mu = np.matmul(np.linalg.inv(B_mu), v)
            coef_alpha = alpha - u_alpha / w_alpha
            return np.concatenate((coef_mu, [coef_alpha]))
        
        elif self.distribution == "ZeroPoisson":
            p = np.exp(np.matmul(self.X, coef[:,0])) / (1 + np.exp(np.matmul(self.X, coef[:,0])))
            p = p.reshape((self.ncells, 1))
            lam = np.exp(np.matmul(self.X, coef[:,1])).reshape((self.ncells, 1))
            params = np.concatenate((p, lam), axis = 1)
            u_p, u_l = self.score_statistic(params)
            w_p, w_l = self.hessian_weights(params)
            
            z_p = np.matmul(self.X, coef[:,0]) + u_p/w_p
            B_p = np.matmul(self.X.T*w_p,self.X)+tau*np.identity(self.ncovs)
            v_p = np.matmul(self.X.T * w_p, z_p)
            coef_p = np.matmul(np.linalg.inv(B_p), v_p).reshape((-1, 1))
            
            z_l = np.matmul(self.X, coef[:,1]) + u_l/w_l
            B_l = np.matmul(self.X.T*w_l,self.X)+tau*np.identity(self.ncovs)
            v_l = np.matmul(self.X.T * w_l, z_l)
            coef_l = np.matmul(np.linalg.inv(B_l), v_l).reshape((-1, 1))
            
            return np.concatenate((coef_p, coef_l), axis = 1)
        
        elif self.distribution == "GeneralDispersion":
            mu = np.exp(np.matmul(self.X, coef))
            h = 0.1
            l = 2*np.pi/h
            counts = np.zeros(int(l)+1)
            counts_est = np.zeros(int(l)+1)
            var_est = np.zeros(int(l)+1)

            for i in range(int(l)+1):
                sub_col = self.distribution[np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()]
                sub_est = mu[np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()]
                counts[i] = np.mean(sub_col)
                counts_est[i] = np.mean(sub_est)
                var_est[i] = np.mean((sub_col-sub_est)**2)
            
            a, b, c = np.polyfit(counts_est, var_est, 2)
            alpha = np.zeros(self.ncells)
            for i in range(int(l)+1):
                indices = np.logical_and(phase > i*h, phase < (i+1)*h).squeeze()
                alpha[indices] = (b-1)/counts_est[i] + a + c/counts_est[i]**2
            
            params = np.concatenate((mu.reshape(-1,1), alpha.reshape((-1, 1))), axis = 1)
            u_mu = self.score_statistic(params)
            w_mu = self.hessian_weights(params)           
            z_mu = np.matmul(self.X, coef) + u_mu/w_mu
            B_mu = np.matmul(self.X.T*w_mu,self.X)+tau*np.identity(self.ncovs)
            v = np.matmul(self.X.T * w_mu, z_mu)
            coef_mu = np.matmul(np.linalg.inv(B_mu), v)
            return coef_mu
        
        else:
            mu = np.exp(np.matmul(self.X, coef))
            sigma = np.sum((self.response - mu)**2) / self.ncells
            params = np.concatenate((mu.reshape((-1,1)), sigma*np.ones((self.ncells, 1))), axis = 1)
        
        z = np.matmul(self.X, coef) + self.score_statistic(params)/self.hessian_weights(params)
        B = np.matmul(self.X.T*self.hessian_weights(params),self.X)+tau*np.identity(self.ncovs)
        v = np.matmul(self.X.T * self.hessian_weights(params), z)
        new_coef = np.matmul(np.linalg.inv(B), v)

        return new_coef.reshape(-1)

    def regularized_fit(self, n_iter, tau):
        
        if self.distribution == "ZeroPoisson":
            coef = np.zeros((self.ncovs, 2))
            coef[0, 0] = 1
            coef[0, 1] = 1
            
        elif self.distribution == "NB":
            coef = np.zeros(self.ncovs+1)
            coef[0] = np.log(np.mean(self.response))
            coef[-1] = 1
            
        else:
            coef = np.zeros(self.ncovs)
            coef[0] = np.log(np.mean(self.response))
        
        for i in range(n_iter):
            coef = self.regularized_iteration(coef, self.ncells*tau)
            
        return coef
    
def fourier_basis(phi, num_harmonics):
    basis = np.ones((len(phi), 1+ 2*num_harmonics))
    for n in range(num_harmonics):
        basis[:, 2*n+1]  = np.cos((n+1)*phi)
        basis[:, 2*n+2]  = np.sin((n+1)*phi)
    return basis



def get_angular_shift(angle1, angle2):
    """
    Calculates the angular shift between two angles in radians,
    ranging from -pi to pi.

    Args:
        angle1 (float): The first angle in radians.
        angle2 (float): The second angle in radians.

    Returns:
        float: The angular shift between angle1 and angle2 in radians,
        ranging from -pi to pi.
    """
    # Normalize angles to be within -pi to pi range
    angle1 = math.atan2(math.sin(angle1), math.cos(angle1))
    angle2 = math.atan2(math.sin(angle2), math.cos(angle2))

    # Calculate the angular shift
    shift = angle2 - angle1

    # Normalize the shift to be within -pi to pi range
    if shift > math.pi:
        shift -= 2 * math.pi
    elif shift < -math.pi:
        shift += 2 * math.pi

    return shift

def scientific_notation_y(ax, decimals=1):
    v = np.asarray(ax.get_yticks().tolist())
    v_abs = np.abs(v)
    v_max = np.max(v_abs)
    exp = 0

    if v_max >= 10:
        sign = '+'
        while v_max >= 10:
            exp = exp + 1
            v_max = v_max / 10
        v = v / 10**exp
    elif v_max <= 1:
        sign = '-'
        while v_max <= 1:
            exp = exp + 1
            v_max = v_max * 10
        v = v * 10**exp
    v = np.around(v, decimals)
    ax.annotate(r'1e' + sign + str(exp), xycoords='axes fraction',
                xy=(0, 0), xytext=(0, 1.01), size=50)
    ax.set_yticklabels(v)
    return



def fourier_basis(phi, num_harmonics):
    basis = np.ones((len(phi), 1+ 2*num_harmonics))
    for n in range(num_harmonics):
        basis[:, 2*n+1]  =  np.cos((n+1)*phi)
        basis[:, 2*n+2]  = np.sin((n+1)*phi)
    return basis


n_harm=1
if(n_harm==1): sd=np.array([.1,.2,.2])[:,None]
if(n_harm==2): sd=np.array([.1,.2,.2,.1,.1])[:,None]

def get_nus(data, phase, noise): 
    # S fit 
    S_pau = data.layers['spliced']
    n_cells, n_genes = S_pau.shape
    X = fourier_basis(np.array(phase).reshape(-1), n_harm)
    logN = np.log(S_pau.sum(1))

    S_est = np.zeros((n_cells, n_genes))
    coef = np.zeros((n_genes, np.shape(X)[1]))
    logl = np.zeros((n_cells, n_genes))
    alphas = np.zeros(n_genes)

    for g in range(n_genes):
        #print(g)
        model = Noise_Model(S_pau[:, g], X, logN , noise)
        if noise=='Poisson':
            
            coef[g,:] = model.fit(50)
            S_est[:,g] = np.exp(np.matmul(X, coef[g, :]))
            logl[:,g] = model.loglikelihood(S_est[:,g])
        if noise=='NB':
            coef[g,:] = model.fit(20)[:-1]
            alphas[g]=coef[g,-1]
            S_est[:,g] = np.exp(np.matmul(X, coef[g, :]))
            #logl[:,g] = model.loglikelihood(S_est[:,g])
            
    # Put estimations in new objects
    phis_pyro = phase
    fourier_coef = coef.T
    fourier_coef_sd = np.zeros(coef.T.shape)
    xs = fourier_coef[1]
    ys = fourier_coef[2]
    r = fourier_coef[0]
    angle = np.arctan2(ys, xs)
    amplitudes = np.sqrt((fourier_coef[1])**2 + (fourier_coef[2])**2 )
    gene_names = data.var.index
    fit_ElogS = S_est.T
    
    # Gene selection 
    print(amplitudes)
    index_s_selected = np.array(amplitudes) > 0.2
    gene_s_selected = np.array(data.var.index)[index_s_selected]
    
    data= data[:,list(gene_s_selected)]
    cell_not_null = data.layers['unspliced'].sum(1)>0
    data = data[cell_not_null]
    
    return [phis_pyro,fourier_coef,fourier_coef_sd,  xs, ys, r,angle, amplitudes, gene_names, fit_ElogS, 
           index_s_selected, gene_s_selected, data, cell_not_null, logN]



def get_nus_unspliced(data, phase, noise, cell_not_null): 
    # U fit 
    U_pau = data.layers['unspliced']
    n_cells, n_genes = U_pau.shape
    X_U = fourier_basis(np.array(phase[cell_not_null]).reshape(-1), n_harm)
    
    logN_U = np.log(U_pau.sum(1))
    U_est = np.zeros((n_cells, n_genes))
    coef_U = np.zeros((n_genes, np.shape(X_U)[1]))
    logl_U = np.zeros((n_cells, n_genes))
    alphas_U = np.zeros(n_genes)
    
    noise='Poisson'
    #noise='NB'
    for g in range(n_genes):
        #print(g)
        model = Noise_Model(U_pau[:, g], X_U,logN_U, noise)
        if noise=='Poisson':
            coef_U[g,:] = model.fit(20)
            U_est[:,g] = np.exp(np.matmul(X_U, coef_U[g, :]))
            logl_U[:,g] = model.loglikelihood(U_est[:,g])
        if noise=='NB':
            coef[g,:] = model.fit(20)[:-1]
            alphas[g]=coef[g,-1]
            S_est[:,g] = np.exp(np.matmul(X, coef[g, :]))
            
    # Put estimations in new objects
    phis_pyro = phase[cell_not_null]
    fourier_coef_U = coef_U.T
    fourier_coef_sd_U = np.zeros(coef_U.T.shape)
    
    gene_names_U = data.var.index
    xs_U = fourier_coef_U[1]
    ys_U = fourier_coef_U[2]
    r_U = fourier_coef_U[0]
    angle_U = np.arctan2(ys_U, xs_U)
    amplitudes_U = np.sqrt((fourier_coef_U[1])**2 + (fourier_coef_U[2])**2 )
    fit_ElogU = U_est.T       

    # Gene selection
    index_u_selected = amplitudes_U > 0.1
    gene_u_selected= np.array(gene_names_U)[index_u_selected]
    data= data[:,list(gene_u_selected)]
    
        
    return [phis_pyro,fourier_coef_U,fourier_coef_sd_U,  xs_U, ys_U, r_U ,angle_U, amplitudes_U, gene_names_U, fit_ElogU, 
           index_u_selected, gene_u_selected, data]
    


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def genes_with_shift_in_quadrant(u_sample, shift, low=0.0, high=np.pi/2):
    """
    Return the subset of genes from u_sample[11] whose corresponding shift is in (low, high).
    """
    genes = np.asarray(u_sample[11])
    shift = np.asarray(shift)
    mask = (shift > low) & (shift < high)
    return genes[mask]


def intersect_gene_lists(list_of_gene_arrays):
    """
    Compute the intersection across all arrays of genes.
    Returns a sorted list for determinism.
    """
    if not list_of_gene_arrays:
        return []
    gene_intersection = set(list_of_gene_arrays[0])
    for arr in list_of_gene_arrays[1:]:
        gene_intersection &= set(arr)
        if not gene_intersection:
            break
    return sorted(gene_intersection)


def extract_common_shifts(nu_u_infos, shifts_sample, common_genes):
    """
    For each sample, gather the shifts corresponding to the common_genes.
    Returns a list of arrays: one array per sample (all arrays have the same length).
    """
    common_genes = list(common_genes)
    commun_shifts = []

    for u_sample, shift in zip(nu_u_infos, shifts_sample):
        genes = np.asarray(u_sample[11])
        shift = np.asarray(shift)

        # Map gene -> index once (faster than np.where in a loop)
        gene_to_idx = {g: i for i, g in enumerate(genes)}

        # Ensure all common genes exist in this sample
        idx = [gene_to_idx[g] for g in common_genes if g in gene_to_idx]
        # If any gene is missing in this sample, skip it (shouldn't happen if "common" is correct)
        if len(idx) != len(common_genes):
            # You can raise an error instead if you prefer strictness
            # raise ValueError("Some common genes were not found in a sample.")
            pass

        commun_shifts.append(shift[idx])

    return commun_shifts  # list length == n_samples; each element length == n_common_genes


def make_boxplot(df, title, palette, rotate_xticks=True):
    """
    Draw a seaborn boxplot with consistent styling.
    """
    plt.figure(figsize=(10, 10))
    sns.boxplot(
        data=df,
        notch=True,
        showcaps=False,
        palette=palette,
        flierprops={"marker": "x"},
        boxprops={"alpha": 0.9, "linewidth": 2, "edgecolor": "white"},
    )
    if rotate_xticks:
        plt.xticks(rotation=90, fontsize=15)
    else:
        plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def bootstrap_first_pc_scale(X, n_boot=1000, rng=None):
    """
    Bootstrap the first principal component scale using SVD on resampled rows.
    Returns omega of shape (n_boot, M) where M is the number of columns in X.
    The original logic computes:
      scale = median(u[:,0] * s[0])
      omega[n,:] = abs(scale * vt[0,:]) * log(2)
    """
    X = np.asarray(X)
    N, M = X.shape
    omega = np.zeros((n_boot, M))

    if rng is None:
        rng = np.random.default_rng()

    for n in range(n_boot):
        # Resample rows with replacement
        idx = rng.choice(N, N, replace=True)
        x = X[idx, :]

        # SVD on the bootstrapped sample
        u, s, vt = np.linalg.svd(x, full_matrices=False)

        # Projected scores on PC1 (u[:,0] * s[0]) -> take a robust scale via median
        scale = np.median(u[:, 0] * s[0])

        # Loadings of PC1: vt[0, :]
        omega[n, :] = np.abs(scale * vt[0, :]) * np.log(2)

    return omega

def get_angular_shift(angle1, angle2):
    """
    Calculates the angular shift between two angles in radians,
    ranging from -pi to pi.

    Args:
        angle1 (float): The first angle in radians.
        angle2 (float): The second angle in radians.

    Returns:
        float: The angular shift between angle1 and angle2 in radians,
        ranging from -pi to pi.
    """
    # Normalize angles to be within -pi to pi range
    angle1 = math.atan2(math.sin(angle1), math.cos(angle1))
    angle2 = math.atan2(math.sin(angle2), math.cos(angle2))

    # Calculate the angular shift
    shift = angle2 - angle1

    # Normalize the shift to be within -pi to pi range
    if shift > math.pi:
        shift -= 2 * math.pi
    elif shift < -math.pi:
        shift += 2 * math.pi

    return shift
import math

def make_boxplot_pretty(df, palette=None, rotate_xticks=True):
    """
    Prettier boxplot:
    - only left/bottom axes, thicker
    - no title
    - boxes closer together
    - black, thicker outlines (box, whiskers, caps, median)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(
        data=df,
        width=0.55,                 # narrower boxes -> closer together
        notch=True,
        showcaps=True,
        palette=palette,
        ax=ax,
        flierprops={"marker": "x"},
        boxprops={"alpha": 0.9, "linewidth": 2.5, "edgecolor": "black"},
        whiskerprops={"linewidth": 2.5, "color": "black"},
        capprops={"linewidth": 2.5, "color": "black"},
        medianprops={"linewidth": 2.5, "color": "black"},
    )

    # Only left & bottom spines, thicker and black
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Thicker ticks; optionally rotate x labels
    ax.tick_params(axis="both", width=2.0, length=6)
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.tick_params(labelsize=13)

    # Remove extra horizontal padding so boxes sit closer
    ax.margins(x=0.01)

    # No title
    ax.set_title("")

    fig.tight_layout()
    plt.show()



