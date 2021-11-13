

export Loss, QuadLoss, LogisticLoss, PoissonLoss, NoLoss

abstract type Loss end

###############################
# Quadratic loss (Normal data)
###############################
struct QuadLoss <: Loss 
    scale::Number
end

function compute_quadloss!(Z, A, tau)
    return 0.5*sum( (Z - A).^2 .* transpose(tau))
end

function compute_quadloss_delta!(Z, A)
    Z .-= A
    return nothing
end

###############################
# Logistic loss (binary data)
###############################
struct LogisticLoss <: Loss 
    scale::Number
end

function compute_logloss!(Z, A)
    Z .= exp.(-Z)
    Z .+= 1.0
    Z .= 1.0 ./Z
    return -sum( A .* log.(1e-12 .+ Z) + (1.0 .- A).*log.(1e-12 + 1.0 .- Z) )
end


function compute_logloss_delta!(Z, A)
    Z .= exp.(-Z)
    Z .+= 1.0
    Z .= 1.0 ./ Z
    Z .-= A
    return nothing 
end

###############################
# Poisson loss (count data)
###############################
struct PoissonLoss <: Loss 
    scale::Number
end

function compute_poissonloss!(Z, A)
    if size(Z,2) == 0 
        loss = 0.0
    else
        loss = sum(1.0.*Z.*A - exp.(1.0.*Z))
    end
    return loss
end

function compute_poissonloss_delta!(Z, A)
    Z .= exp.(Z)
    Z .*= -1.0
    Z .+= A
    return nothing
end

##########################################
## Ternary loss (tailored for CNA data)
##########################################
#
#struct TernaryLoss <: Loss
#    scale::Number
#end
#
#function compute_ternaryloss!(Z, A, tau)
#    
#end

##################################
# NoLoss (for latent features)
##################################
struct NoLoss <: Loss 
    scale::Number
end

function compute_noloss!(Z,A)
    return 0.0
end

function compute_noloss_delta!(Z, A)
    return nothing
end

function vprint(s; verbose=false)
    if verbose
        println(s)
    end
end

##################################
# Other functions
##################################
function compute_loss!(X, Y, Z_ql, Z_ll, Z_pl, 
                       A_ql, A_ll, A_pl,
                       beta, beta_reg,
                       mu, mu_reg, theta, theta_reg, 
                       tau, a_0_tau, b_0_tau,
                       X_reg_mats, Y_reg_mats;
                       verbose=false)
    loss = compute_quadloss!(Z_ql, A_ql, tau)
    vprint(string("\tQUAD LOSS: ", loss), verbose=verbose)
    loss += compute_logloss!(Z_ll, A_ll)
    vprint(string("\tLOGISTIC LOSS: ", loss), verbose=verbose)
    loss += compute_poissonloss!(Z_pl, A_pl)
    vprint(string("\tPOISSON LOSS: ", loss), verbose=verbose)
    
    for i=1:size(beta,1)
        loss += 0.5 * dot(beta[i,:], beta_reg*beta[i,:])
    end
    vprint(string("\tBETA LOSS: ", loss), verbose=verbose)
    loss += compute_mat_reg_loss(X, X_reg_mats)
    vprint(string("\tX REG LOSS: ", loss), verbose=verbose)
    loss += compute_mat_reg_loss(Y, Y_reg_mats)
    vprint(string("\tY REG LOSS: ", loss), verbose=verbose)
    loss += compute_vec_reg_loss(mu, mu_reg)
    vprint(string("\tMU REG LOSS: ", loss), verbose=verbose)
    loss += compute_vec_reg_loss(theta, theta_reg)
    vprint(string("\tTHETA REG LOSS: ", loss), verbose=verbose)

    M = size(X,2)
    loss += compute_tau_loss(tau, a_0_tau, b_0_tau, M) 
    vprint(string("\tTAU LOSS: ", loss), verbose=verbose)
    
    return loss
end


function compute_tau_loss(tau, a_0_tau, b_0_tau, M)
    return sum((b_0_tau.*tau) .- (0.5*M + a_0_tau - 1).*log.(tau)) 
end


function compute_grad_delta!(Z_ql, Z_ll, Z_pl, 
                             A_ql, A_ll, A_pl)
    compute_quadloss_delta!(Z_ql, A_ql)
    compute_logloss_delta!(Z_ll, A_ll)
    compute_poissonloss_delta!(Z_pl, A_pl)
end


function compute_grad_X!(grad_X, X, Y, Z, 
                         Z_ql, Z_ll, Z_pl, 
                         A_ql, A_ll, A_pl, 
                         inst_reg_mats)

    compute_grad_delta!(Z_ql, Z_ll, Z_pl, 
                        A_ql, A_ll, A_pl)

    N = size(Z,2)
    grad_X .= (Y * transpose(Z)) 

    add_mat_reg_grad!(grad_X, X, inst_reg_mats) 
end


function compute_grad_Y!(grad_Y, X, Y, Z, 
                         Z_ql, Z_ll, Z_pl, 
                         A_ql, A_ll, A_pl, 
                         feat_reg_mats)

    compute_grad_delta!(Z_ql, Z_ll, Z_pl, 
                        A_ql, A_ll, A_pl)

    M = size(Z,1)
    grad_Y .= X * Z 

    add_mat_reg_grad!(grad_Y, Y, feat_reg_mats) 
end


function compute_grad_beta!(grad_beta, beta, C, Z, tau, beta_reg)
    grad_beta .= (transpose(C)*Z) .* transpose(tau)
    grad_beta .+= transpose(beta_reg*transpose(beta))
end


function compute_grad_mu!(grad_mu, mu, Z, tau, mu_reg_mat)
    grad_mu .= Z*tau
    grad_mu .+= mu_reg_mat*mu
end

function compute_grad_theta!(grad_theta, theta, Z, tau, theta_reg_mat)
   grad_theta .= (sum(Z, dims=1)[1,:] .* tau)
   grad_theta .+= theta_reg_mat*theta
end

function update_tau!(tau, Z, a_0, b_0)
    M = size(Z, 1)
    tau .= (2.0*a_0 + M)./(2.0*b_0 .+ sum(Z.^2.0, dims=1)[1,:]) 
end

###########################
# PRECOMPILE
gpu_mat = CuArray{Float32,2}
gpu_vec = CuArray{Float32,1}
gpu_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}

precompile(compute_quadloss!, (gpu_mat, gpu_mat, gpu_vec))
precompile(compute_logloss!, (gpu_mat, gpu_mat))
precompile(compute_poissonloss!, (gpu_mat, gpu_mat))

precompile(compute_tau_loss, (gpu_vec, Float32, Float32, Int64))

precompile(compute_loss!, (gpu_mat, gpu_mat, 
                           gpu_mat, gpu_mat, gpu_mat,
                           gpu_mat, gpu_mat, gpu_mat,
                           gpu_mat, gpu_sparse, 
                           gpu_mat, gpu_sparse,
                           gpu_vec, gpu_sparse, gpu_vec, gpu_sparse, 
                           gpu_vec, Float32, Float32,
                           Vector{gpu_sparse}, Vector{gpu_sparse}))


precompile(compute_grad_delta!, (gpu_mat, gpu_mat, gpu_mat, 
                                 gpu_mat, gpu_mat, gpu_mat))
precompile(compute_quadloss_delta!, (gpu_mat, gpu_mat))
precompile(compute_logloss_delta!, (gpu_mat, gpu_mat))
precompile(compute_poissonloss_delta!, (gpu_mat, gpu_mat))
