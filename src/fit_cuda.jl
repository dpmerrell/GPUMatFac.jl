

import ScikitLearnBase: fit!

export fit!, fit_line_search!


function fit!(model::MatFacModel, A::AbstractMatrix; method="nesterov", kwargs...)

    if method == "nesterov"
        fit_nesterov!(model, A; kwargs...)
    elseif method == "line_search"
        fit_line_search!(model, A; kwargs...)
    end

end

function compute_Z!(Z, X, Y, mu, theta)
    Z .= transpose(X)*Y
    Z .+= mu
    Z .+= transpose(theta)
end


function nesterov_update!(value, velocity, grad, momentum, lr)
   
   # scale gradient by learning rate
   grad .*= lr

   # update velocity
   velocity .*= momentum
   velocity .-= grad

   # complete value update
   value .-= grad

   # take momentum "half-step"
   value .+= (momentum.*velocity)

   return value, velocity

end

function print_array_summary(array, name)

    println(name, ": ", size(array))
    println("\tmin:", minimum(array)) 
    println("\tmax:", maximum(array)) 
    println("\tmean:", sum(array)/prod(size(array))) 
end

function fit_nesterov!(model::MatFacModel, A::AbstractMatrix;
                       inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
                       a_0_tau::Real=1.0, b_0_tau::Real=1e-3,
                       max_iter::Integer=100, 
                       lr::Real=0.1, momentum::Real=0.5,
                       abs_tol::Real=1e-3, rel_tol::Real=1e-7,
                       loss_iter::Integer=10)

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    @assert size(model.X,1) == size(model.Y,1)

    lr = Float32(lr)
    lr_row = lr / N
    lr_col = lr / M

    # Move data and model to the GPU.
    # Float32 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    mu_d = CuArray{Float32}(model.instance_offset)
    mu_reg_d = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(inst_reg_weight .* model.instance_offset_reg)
    theta_d = CuArray{Float32}(model.feature_offset)
    theta_reg_d = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(feat_reg_weight .* model.feature_offset_reg)
    tau_d = CuArray{Float32}(model.feature_precision)

    A_d = CuArray{Float32}(A)
    Z_d = CUDA.zeros(Float32, M, N)
 
    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float32(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float32(0.5) : Float32(x)
    map!(mask_func, A_d, A_d)

    ## Scaling factors for the columns
    #scales = [loss.scale for loss in model.losses]
    #feature_scales = CuArray{Float32}(transpose(scales))
    
    # Bookkeeping for the loss functions
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

    # Arrays for holding gradients and velocities
    grad_X = CUDA.zeros(Float32, (K, M))
    vel_X = CUDA.zeros(Float32, (K, M))

    grad_mu = CUDA.zeros(Float32, M)
    vel_mu = CUDA.zeros(Float32, M) 
    
    grad_Y = CUDA.zeros(Float32, (K, N))
    vel_Y = CUDA.zeros(Float32, (K, N))

    grad_theta = CUDA.zeros(Float32, N)
    vel_theta = CUDA.zeros(Float32, N)

    while iter < max_iter 

        ################################
        # Update Row quantities (X, mu)

        compute_Z!(Z_d, X_d, Y_d, mu_d, theta_d)
        Z_d .*= obs_mask
        Z_d .+= missing_mask

        # update X (NOTE: this stores Z-A in Z_d)
        compute_grad_X!(grad_X, X_d, Y_d, Z_d, 
                        Z_ql_view, Z_ll_view, Z_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        inst_reg_mats_d)
        X_d, vel_X = nesterov_update!(X_d, vel_X, grad_X, momentum, lr_row)

        # Update mu (Reuse the Z-A stored in Z_d)
        compute_grad_mu!(grad_mu, mu_d, Z_d, tau_d, mu_reg_d)
        mu_d, vel_mu = nesterov_update!(mu_d, vel_mu, grad_mu, momentum, lr_row)

        ############################
        # Update Column quantities (Y, theta, tau)

        compute_Z!(Z_d, X_d, Y_d, mu_d, theta_d)
        Z_d .*= obs_mask
        Z_d .+= missing_mask

        # Update Y
        compute_grad_Y!(grad_Y, X_d, Y_d, Z_d, 
                        Z_ql_view, Z_ll_view, Z_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feat_reg_mats_d)
        Y_d, vel_Y = nesterov_update!(Y_d, vel_Y, grad_Y, momentum, lr_col)

        # Update theta
        compute_grad_theta!(grad_theta, theta_d, Z_d, tau_d, theta_reg_d) 
        theta_d, vel_theta = nesterov_update!(theta_d, vel_theta, grad_theta, momentum, lr_col)

        # Update tau (closed-form exact update)
        update_tau!(tau_ql_view, Z_ql_view, a_0_tau, b_0_tau)

        ############################
        # Every so often, compute the loss
        
        iter += 1
        print_str = "Iteration: $iter"
        
        if (iter % loss_iter == 0)
            
            compute_Z!(Z_d, X_d, Y_d, mu_d, theta_d)
            Z_d .*= obs_mask
            Z_d .+= missing_mask
            
            new_loss = compute_loss!(X_d, Y_d, Z_ql_view, Z_ll_view, Z_pl_view,
                                               A_ql_view, A_ll_view, A_pl_view,
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

    # Move model parameters back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
    model.instance_offset = Array{Float32}(mu_d)
    model.feature_offset = Array{Float32}(theta_d)
    model.feature_precision = Array{Float32}(tau_d)

    return model 
end


function fit_line_search!(model::MatFacModel, A::AbstractMatrix;
                          inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
                          max_iter::Integer=100,
                          alpha=1.0, c1=1e-5, c2=0.9, grow=1.5, shrink=0.5, 
                          line_search_max_iter=10,
                          abs_tol::Real=1e-3, rel_tol::Real=1e-7)

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    @assert size(model.X,1) == size(model.Y,1)

#    # Distinguish between (1) K, the hidden dimension;
#    #                     (2) the subsets 1:K_opt_X, 1:K_opt_Y of 1:K that is optimized;
#    #                     (3) the subset of optimized dims that are regularized.
#    # The regularized dim is determined by the lists of matrices
#    # provided during MatFacModel construction.
#    # K_opt_X, K_opt_Y have default value K.
#    if K_opt_X == nothing
#        K_opt_X = K
#    end
#    if K_opt_Y == nothing
#        K_opt_Y = K
#    end
#    @assert K_opt_X <= K
#    @assert K_opt_Y <= K

    # Move data and model to the GPU.
    # Float32 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float32}(A)
    XY_d = CUDA.zeros(Float32, M,N)
 
    # Some frequently-used views of the X and Y arrays
    #X_d_opt_view = view(X_d, 1:K_opt_X, :)
    #X_d_grady_view = view(X_d, 1:K_opt_Y, :)

    #Y_d_opt_view = view(Y_d, 1:K_opt_Y, :)
    #Y_d_gradx_view = view(Y_d, 1:K_opt_X, :)

    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float32(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float32(0.5) : Float32(x)
    map!(mask_func, A_d, A_d)

    # Scaling factors for the columns
    scales = [loss.scale for loss in model.losses]
    feature_scales = CuArray{Float32}(transpose(scales))
    
    # Bookkeeping for the loss functions
    ql_idx = CuVector{Int64}(findall(typeof.(model.losses) .== QuadLoss))
    ll_idx = CuVector{Int64}(findall(typeof.(model.losses) .== LogisticLoss))
    pl_idx = CuVector{Int64}(findall(typeof.(model.losses) .== PoissonLoss))

    XY_ql_view = view(XY_d, :, ql_idx)
    XY_ll_view = view(XY_d, :, ll_idx)
    XY_pl_view = view(XY_d, :, pl_idx)
    A_ql_view = view(A_d, :, ql_idx)
    A_ll_view = view(A_d, :, ll_idx)
    A_pl_view = view(A_d, :, pl_idx)

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CUDA.CUSPARSE.CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients
    grad_X = CUDA.zeros(Float32, (K, M))
    grad_Y = CUDA.zeros(Float32, (K, N))

    # These functions will be accepted as arguments
    # to the line search procedure
    function grad_fn_x(X_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        compute_grad_X!(grad_X, X_d, Y_d, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        inst_reg_mats_d)
        return grad_X
    end

    function grad_fn_y(Y_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        compute_grad_Y!(grad_Y, X_d, Y_d, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        feat_reg_mats_d)
        return grad_Y
    end

    function loss_fn_x(X_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        return compute_loss!(X_d, Y_d, XY_ql_view, XY_ll_view, XY_pl_view,
                             A_ql_view, A_ll_view, A_pl_view,
                             inst_reg_mats_d, feat_reg_mats_d)
    end

    function loss_fn_y(Y_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        return compute_loss!(X_d, Y_d, XY_ql_view, XY_ll_view, XY_pl_view,
                             A_ql_view, A_ll_view, A_pl_view,
                             inst_reg_mats_d, feat_reg_mats_d)
    end

    alpha_x = alpha
    alpha_y = alpha

    while iter < max_iter 

        ############################
        # Update X 
        alpha_x, _ = grad_descent_line_search!(X_d, grad_X, 
                                               loss_fn_x, grad_fn_x;
                                               alpha=alpha_x, 
                                               c1=c1, c2=c2, 
                                               grow=grow, shrink=shrink, 
                                               max_iter=line_search_max_iter)


        ############################
        # Update Y
        alpha_y, new_loss = grad_descent_line_search!(Y_d, grad_Y, 
                                                      loss_fn_y, grad_fn_y;
                                                      alpha=alpha_y, 
                                                      c1=c1, c2=c2, 
                                                      grow=grow, shrink=shrink, 
                                                      max_iter=line_search_max_iter)

        ############################
        # Print some information
        iter += 1
        print_str = "Iteration: $iter"

        print_str = string(print_str, "\tLoss: $new_loss")
        println(print_str)

        ############################
        # Check termination conditions
        if abs(new_loss - cur_loss) < abs_tol
            println(string("Absolute change <", abs_tol, ". Terminating."))
            break
        end
        if abs((new_loss - cur_loss)/cur_loss) < rel_tol
            println(string("Relative change <", rel_tol, ". Terminating."))
            break
        end
        cur_loss = new_loss


    end # while

    # Move model X and Y back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
   
    return model 
end



