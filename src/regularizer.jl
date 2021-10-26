

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

function compute_vec_reg_loss(x::CuArray,
                              reg_mat::AbstractMatrix)
    return 0.5 * dot(x, reg_mat*x)
end


