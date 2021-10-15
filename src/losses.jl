

export Loss, QuadLoss, LogisticLoss, PoissonLoss, NoLoss

abstract type Loss end

###############################
# Quadratic loss (Normal data)
###############################
struct QuadLoss <: Loss 
    scale::Number
end

function evaluate(ql::QuadLoss, x, y, a)
    return ql.scale * 0.5 * (dot(x,y) - a)^2
end

function grad_x(ql::QuadLoss, x, y, a)
    return ql.scale * (dot(x,y) - a) .* y
end

function grad_y(ql::QuadLoss, x, y, a)
    return ql.scale * (dot(x,y) - a) .* x
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

function evaluate(ll::LogisticLoss, x, y, a)
    z = dot(x,y)
    return ll.scale * ( log(1.0 + exp(-z)) + (1.0-a)*z )
end

function accum_grad_x!(g, ll::LogisticLoss, y, xy, a)
    BLAS.axpy!(ll.scale * ( (1.0-a) - 1.0/(1.0 + exp(xy)) ), y, g)
    return
end

function accum_grad_y!(g, ll::LogisticLoss, x, xy, a)
    BLAS.axpy!(ll.scale * ( (1.0-a) - 1.0/(1.0 + exp(xy)) ), x, g)
    return 
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

function evaluate(pl::PoissonLoss, x, y, a)
    z = dot(x,y)
    return pl.scale * ( a*z - exp(z) )
end

function grad_x(pl::PoissonLoss, x, y, a)
    return pl.scale * ( a - exp(dot(x,y)) ) .* y
end

function grad_y(pl::PoissonLoss, x, y, a)
    return pl.scale * ( a - exp(dot(x,y)) ) .* x
end

function compute_poissonloss!(Z, A)
    return sum(1.0.*Z.*A - exp.(1.0.*Z))
end

function compute_poissonloss_delta!(Z, A)
    Z .= exp.(Z)
    Z .*= -1.0
    Z .+= A
    return nothing
end


##################################
# NoLoss (for latent features)
##################################
struct NoLoss <: Loss 
    scale::Number
end

function evaluate(nl::NoLoss, x, y, a)
    return 0.0
end

function grad_x(Z, A)

end

function compute_noloss!(Z,A)
    return 0.0
end

function compute_noloss_delta!(Z, A)
    return nothing
end


##################################
# Other functions
##################################
function compute_loss!(X, Y, Z_ql, Z_ll, Z_pl, 
                       A_ql, A_ll, A_pl,
                       mu, mu_reg, theta, theta_reg, 
                       tau, a_0_tau, b_0_tau,
                       X_reg_mats, Y_reg_mats)
    loss = compute_quadloss!(Z_ql, A_ql, tau)
    loss += compute_logloss!(Z_ll, A_ll)
    loss += compute_poissonloss!(Z_pl, A_pl)
    loss += compute_mat_reg_loss(X, X_reg_mats)
    loss += compute_mat_reg_loss(Y, Y_reg_mats)
    loss += compute_vec_reg_loss(mu, mu_reg)
    loss += compute_vec_reg_loss(theta, theta_reg)

    M = size(X,2)
    loss += compute_tau_loss(tau, a_0_tau, b_0_tau, M) 
    
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


function compute_grad_mu!(grad_mu, mu, Z, tau, mu_reg_mat)
    grad_mu .= Z*tau
    add_vec_reg_grad!(grad_mu, mu, mu_reg_mat)
end

function compute_grad_theta!(grad_theta, theta, Z, tau, theta_reg_mat)
   grad_theta .= (sum(Z, dims=1)[1,:] .* tau)
   add_vec_reg_grad!(grad_theta, theta, theta_reg_mat)
end

function update_tau!(tau, Z, a_0, b_0)
    M = size(Z, 1)
    tau .= (2.0*a_0 + M)./(2.0*b_0 .+ sum(Z.^2.0, dims=1)[1,:]) 
end

