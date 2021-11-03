
using LinearAlgebra, CUDA, SparseArrays

export MatFacModel

mutable struct MatFacModel

    X::AbstractMatrix            # KxM "instance factor" matrix
    Y::AbstractMatrix            # KxN "feature factor" matrix

    instance_reg_mats::AbstractVector{AbstractMatrix}  # K x (M x M) (usually sparse)
    feature_reg_mats::AbstractVector{AbstractMatrix}  # K x (N x N) (usually sparse)

    instance_offset::AbstractVector # instance-wise "intercept" terms
    instance_offset_reg::AbstractMatrix # regularizer matrix (usually sparse)

    feature_offset::AbstractVector  # feature-wise "intercept" terms
    feature_offset_reg::AbstractMatrix # regularizer matrix (usually sparse)

    feature_precision::AbstractVector # feature-specific precisions

    instance_covariate_coeff::Union{Nothing,AbstractMatrix} # Regression coefficients 
                                                          # for sample covariates
    instance_covariate_coeff_reg::AbstractMatrix
    
    losses::AbstractVector       # N-dim vector of feature-specific losses

    factor_importances::AbstractVector # K-dim vector of factor importances
end


function MatFacModel(instance_reg_mats::AbstractVector, 
                     feature_reg_mats::AbstractVector,
                     losses::AbstractVector;
                     instance_offset_reg::Union{Nothing,AbstractMatrix}=nothing,
                     feature_offset_reg::Union{Nothing,AbstractMatrix}=nothing,
                     instance_covariate_coeff_reg::Union{Nothing,AbstractMatrix}=nothing,
                     K::Union{Nothing,Integer}=nothing)

    M = size(instance_reg_mats[1],1)
    N = size(feature_reg_mats[1],1)

    if K == nothing
        K = max(length(instance_reg_mats),
                length(feature_reg_mats))
    else
        @assert K >= max(length(instance_reg_mats),
                         length(feature_reg_mats))
    end

    X = 0.01 .* randn(K, M) ./ sqrt(K) 
    Y = 0.01 .* randn(K, N) 

    instance_offset = 0.01 .* randn(M)
    feature_offset = 0.01 .* randn(N)

    if instance_offset_reg == nothing
        instance_offset_reg = spzeros(M,M)
    end
    if feature_offset_reg == nothing
        feature_offset_reg = spzeros(N,N)
    end
    if instance_covariate_coeff_reg == nothing
        instance_covariate_coeff_reg = spzeros(N,N)
    end

    feature_precision = ones(N)

    instance_covariate_coeff = nothing

    factor_importances = zeros(K)

    return MatFacModel(X, Y, instance_reg_mats,
                             feature_reg_mats,
                             instance_offset,
                             instance_offset_reg,
                             feature_offset,
                             feature_offset_reg,
                             feature_precision,
                             instance_covariate_coeff,
                             instance_covariate_coeff_reg,
                             losses,
                             factor_importances
                      )
end


