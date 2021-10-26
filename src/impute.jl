
export impute_values


function logistic(x)
    return 1.0 ./ (1.0 + exp.(-x))
end


function compute_Z(model::MatFacModel;
                   covariates::Union{Nothing,Matrix}=nothing,
                   factor_set::Union{Nothing,Vector{Int64}}=nothing)

    if factor_set == nothing
        factor_set = collect(1:size(model.X,1))
    end
    X = model.X[factor_set,:]
    Y = model.Y[factor_set,:]

    Z = transpose(X)*Y
    Z .+= model.instance_offset
    Z .+= transpose(model.feature_offset)

    if covariates != nothing
        Z .+= covariates*model.instance_covariate_coeff
    end

    return Z
end


function apply_link_function(Z::Matrix, model::MatFacModel)

    logistic_idx = (model.losses .<: LogisticLoss)
    poisson_idx = (model.losses .<: PoissonLoss)

    Z[:,logistic_idx] .= logistic(Z[:,logistic_idx])
    Z[:,poisson_idx] .= exp.(Z[:,poisson_idx])

    return Z
end


function impute_values(model::MatFacModel; 
                       covariates::Union{Nothing,Matrix}=nothing,
                       factor_set::Union{Nothing,Vector{Int64}}=nothing)
    
    Z = compute_Z(model; covariates=covariates, factor_set=factor_set)
    Z = apply_link_function(Z, model)
    return Z
end


