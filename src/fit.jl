

import ScikitLearnBase: fit!

export fit!, fit_line_search!


function fit!(model::MatFacModel, A::AbstractMatrix; method="adagrad", kwargs...)

    if method == "adagrad"
        fit_adagrad!(model, A; kwargs...)
    elseif method == "line_search"
        fit_line_search!(model, A; kwargs...)
    end

end

function compute_Z!(Z, X, Y, C, beta, mu, theta)
    Z .= transpose(X)*Y
    Z .+= C*beta
    Z .+= mu
    Z .+= transpose(theta)
end


function adagrad_update!(value, grad, G, lr)
   
   # complete value update
   G .+= (grad.^2.0)

   value .-= grad .* (lr ./ sqrt.(G))
   
   return value, G 

end


function normalize_factors!(X,Y)

    N = size(Y, 2)
    target_size = N/10.0 # This is kind of arbitrary
    Y_norms = sqrt.(sum(Y.^2, dims=2))

    multipliers = target_size ./ Y_norms
    Y .*= multipliers
    X ./= multipliers
end


function fit_adagrad!(model::MatFacModel, A::AbstractMatrix;
                      instance_covariates::Union{Nothing,AbstractMatrix}=nothing,
                      inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
                      a_0_tau::Real=1.0, b_0_tau::Real=1e-3,
                      max_iter::Integer=1000, 
                      lr::Real=0.01, eps::Real=1e-8,
                      abs_tol::Real=1e-3, rel_tol::Real=1e-7,
                      loss_iter::Integer=10,
                      normalize_Y::Bool=true)

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    @assert size(model.X,1) == size(model.Y,1)

    lr = Float32(lr)
    lr_row = lr #/ N
    lr_col = lr #/ M

    if instance_covariates == nothing
        instance_covariates = zeros(M,1)
        instance_covariate_coeff = zeros(1,N)
    else
        @assert size(instance_covariates,1) == M
        instance_covariate_coeff = randn(size(instance_covariates,2), N)
    end

    covariate_dim = size(instance_covariates, 2)

    # Move data and model to the GPU.
    # Float32 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    beta_d = CuArray{Float32}(instance_covariate_coeff)
    beta_reg_d = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(feat_reg_weight .* model.instance_covariate_coeff_reg)

    mu_d = CuArray{Float32}(model.instance_offset)
    mu_reg_d = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(inst_reg_weight .* model.instance_offset_reg)
    theta_d = CuArray{Float32}(model.feature_offset)
    theta_reg_d = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(feat_reg_weight .* model.feature_offset_reg)
    tau_d = CuArray{Float32}(model.feature_precision)

    A_d = CuArray{Float32}(A)
    C_d = CuArray{Float32}(instance_covariates)
    Z_d = CUDA.zeros(Float32, M, N)
 
    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float32(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float32(0.5) : Float32(x)
    map!(mask_func, A_d, A_d)

    # Bookkeeping for the different noise models/loss functions
    ql_idx = CuVector{Int64}(findall(typeof.(model.losses) .== QuadLoss))
    ll_idx = CuVector{Int64}(findall(typeof.(model.losses) .== LogisticLoss))
    pl_idx = CuVector{Int64}(findall(typeof.(model.losses) .== PoissonLoss))

    Z_ql_view = view(Z_d, :, ql_idx)
    Z_ll_view = view(Z_d, :, ll_idx)
    Z_pl_view = view(Z_d, :, pl_idx)
    A_ql_view = view(A_d, :, ql_idx)
    A_ll_view = view(A_d, :, ll_idx)
    A_pl_view = view(A_d, :, pl_idx)

    tau_ql_view = view(tau_d, ql_idx)

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients and summed gradients 
    grad_X = CUDA.zeros(Float32, (K, M))
    G_X = CuArray{Float32}(fill(eps, K, M))

    grad_mu = CUDA.zeros(Float32, M)
    G_mu = CuArray{Float32}(fill(eps, M)) 
    
    grad_Y = CUDA.zeros(Float32, (K, N))
    G_Y = CuArray{Float32}(fill(eps, K, N))

    grad_beta = CUDA.zeros(Float32, (covariate_dim, N))
    G_beta = CuArray{Float32}(fill(eps, covariate_dim, N))

    grad_theta = CUDA.zeros(Float32, N)
    G_theta = CuArray{Float32}(fill(eps, N))

    while iter < max_iter 

        ################################
        # Update Row quantities (X, mu)

        compute_Z!(Z_d, X_d, Y_d, C_d, beta_d, mu_d, theta_d)
        Z_d .*= obs_mask
        Z_d .+= missing_mask

        # update X (NOTE: this stores Z-A in Z_d)
        compute_grad_X!(grad_X, X_d, Y_d, Z_d, 
                        Z_ql_view, Z_ll_view, Z_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        inst_reg_mats_d)
        X_d, G_X = adagrad_update!(X_d, grad_X, G_X, lr_row)

        # Update mu (Reuse the Z-A stored in Z_d)
        compute_grad_mu!(grad_mu, mu_d, Z_d, tau_d, mu_reg_d)
        mu_d, G_mu = adagrad_update!(mu_d, grad_mu, G_mu, lr_row)

        ############################
        # Update Column quantities (Y, beta, theta, tau)

        compute_Z!(Z_d, X_d, Y_d, C_d, beta_d, mu_d, theta_d)
        Z_d .*= obs_mask
        Z_d .+= missing_mask

        # Update Y
        compute_grad_Y!(grad_Y, X_d, Y_d, Z_d, 
                        Z_ql_view, Z_ll_view, Z_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feat_reg_mats_d)
        Y_d, G_Y = adagrad_update!(Y_d, grad_Y, G_Y, lr_col)

        # Update beta
        compute_grad_beta!(grad_beta, beta_d, C_d, Z_d, tau_d, beta_reg_d)
        beta_d, G_beta = adagrad_update!(beta_d, grad_beta, G_beta, lr_col)

        # Update theta
        compute_grad_theta!(grad_theta, theta_d, Z_d, tau_d, theta_reg_d) 
        theta_d, G_theta = adagrad_update!(theta_d, grad_theta, G_theta, lr_col)

        # Update tau (closed-form exact update)
        update_tau!(tau_ql_view, Z_ql_view, a_0_tau, b_0_tau)

        ############################
        # Every so often, compute the loss
        
        iter += 1
        print_str = "Iteration: $iter"
        
        if (iter % loss_iter == 0)
            
            compute_Z!(Z_d, X_d, Y_d, C_d, beta_d, mu_d, theta_d)
            Z_d .*= obs_mask
            Z_d .+= missing_mask
            
            new_loss = compute_loss!(X_d, Y_d, Z_ql_view, Z_ll_view, Z_pl_view,
                                               A_ql_view, A_ll_view, A_pl_view,
                                               beta_d, beta_reg_d,
                                               mu_d, mu_reg_d, theta_d, theta_reg_d, 
                                               tau_ql_view, a_0_tau, b_0_tau,
                                               inst_reg_mats_d, feat_reg_mats_d)
            print_str = string(print_str, "\tLoss: $new_loss")
            println(print_str)

            if abs(new_loss - cur_loss) < abs_tol
                println(string("Absolute change <", abs_tol, ". Terminating."))
                break
            end
            if abs((new_loss - cur_loss)/cur_loss) < rel_tol
                println(string("Relative change <", rel_tol, ". Terminating."))
                break
            end
            cur_loss = new_loss
        end

    end # while

    if normalize_Y
        normalize_factors!(X_d, Y_d)
    end

    # Move model parameters back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
    model.instance_covariate_coeff = Array{Float32}(beta_d)
    model.instance_offset = Array{Float32}(mu_d)
    model.feature_offset = Array{Float32}(theta_d)
    model.feature_precision = Array{Float32}(tau_d)

    return model 
end


# Some precompilation
precompile(compute_Z!, (CuArray{Float32,2}, CuArray{Float32,2}, 
                        CuArray{Float32,2}, CuArray{Float32,2},
                        CuArray{Float32,2}, 
                        CuArray{Float32,1}, CuArray{Float32,1}))

for t in (CuArray{Float32,2}, CuArray{Float32,1})
    precompile(adagrad_update!, (t, t, t, Float32))
end

precompile(normalize_factors!, (CuArray{Float32}, CuArray{Float32}))

