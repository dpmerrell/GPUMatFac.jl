

function reg_eval(X, idx, reg_mats)
    s = 0.0
    for k=1:size(X,1)
        s += X[k, idx] * dot(view(reg_mats[k], :, idx), view(X, k, :))
    end
    return 0.5*s
end


function reg_grad(X, idx, reg_mats, K_opt)
    g = zeros(K_opt)
    for k=1:min(K_opt,length(reg_mats))
        g[k] = X[k, idx]*reg_mats[k][idx,idx] + dot(view(reg_mats[k], :, idx), view(X, k, :))
    end
    return 0.5 .* g
end


#####################################
# CUDA functions

function compute_mat_reg_loss(X::CuArray{Float32,2}, 
                              reg_mats::AbstractVector)
    s = 0.0
    for i=1:length(reg_mats)
        s += 0.5 * dot(X[i,:], reg_mats[i]*X[i,:])
    end
    return s
end

function add_mat_reg_grad!(grad_X::CuArray{Float32,2}, 
                           X::CuArray{Float32,2}, 
                           reg_mats::AbstractVector)
    for i=1:min(length(reg_mats), size(grad_X,1))
        grad_X[i,:] .+= (reg_mats[i]*X[i,:])
    end
    return nothing
end

function compute_vec_reg_loss(x::CuArray{Float32,1},
                              reg_mat::AbstractMatrix)
    return 0.5 * dot(x, reg_mat*x)
end

function add_vec_reg_grad!(grad_x::CuArray{Float32,1},
                           x::CuArray{Float32,1},
                           reg_mat::AbstractMatrix)
    grad_x .+= reg_mat*x
end

