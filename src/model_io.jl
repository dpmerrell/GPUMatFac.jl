
using HDF5

export save_hdf, load_hdf

function save_hdf(hdf_file_path, model; group_name::String="")
    h5open(hdf_file_path, "w") do file
        to_hdf(file, group_name, model)
    end
end

function load_hdf(hdf_file_path; group_name::String="")
    model = h5open(hdf_file_path, "r") do file
        matfac_from_hdf(file, group_name)
    end
    return model
end


####################################
# MatFacModel
####################################

function to_hdf(hdf_file, path::String, model::MatFacModel)

    # Factors
    write(hdf_file, string(path,"/X"), model.X)
    write(hdf_file, string(path,"/Y"), model.Y)

    # covariate coefficients
    if model.instance_covariate_coeff != nothing
        write(hdf_file, string(path,"/instance_covariate_coeff"), 
                        model.instance_covariate_coeff)
    end
    to_hdf(hdf_file, string(path,"/instance_covariate_coeff_reg"),
                     model.instance_covariate_coeff_reg)

    # Offsets
    write(hdf_file, string(path,"/instance_offset"), model.instance_offset)
    write(hdf_file, string(path,"/feature_offset"), model.feature_offset)

    # Offset Regularizers 
    to_hdf(hdf_file, string(path,"/instance_offset_reg"), model.instance_offset_reg)
    to_hdf(hdf_file, string(path,"/feature_offset_reg"), model.feature_offset_reg)

    # Precisions
    write(hdf_file, string(path,"/feature_precision"), model.feature_precision)

    # Losses
    loss_strs = String[string(typeof(loss)) for loss in model.losses]
    write(hdf_file, string(path, "/loss_types"), loss_strs)
    loss_scales = Float64[loss.scale for loss in model.losses]
    write(hdf_file, string(path, "/loss_scales"), loss_scales)

    # Instance regularizer matrices
    for (i, regmat) in enumerate(model.instance_reg_mats)
        to_hdf(hdf_file, string(path, "/inst_reg/mat_", i), regmat)
    end
   
    # Feature regularizer matrices
    for (i, regmat) in enumerate(model.feature_reg_mats)
        to_hdf(hdf_file, string(path, "/feat_reg/mat_", i), regmat)
    end

    write(hdf_file, string(path,"/factor_importances"), model.factor_importances)

end


function matfac_from_hdf(hdf_file, path::String)
    # Factors
    X = hdf_file[string(path,"/X")][:,:]
    Y = hdf_file[string(path,"/Y")][:,:]

    # Covariate coefficients
    if "instance_covariate_coeff" in keys(hdf_file[string(path,"/")])
        covariate_coeff = hdf_file[string(path,"/instance_covariate_coeff")][:,:]
    end
    covariate_coeff_reg = spmat_from_hdf(hdf_file, string(path,"/instance_covariate_coeff_reg"))

    # Offsets
    instance_offset = hdf_file[string(path,"/instance_offset")][:]
    feature_offset = hdf_file[string(path,"/feature_offset")][:]

    # Offset Regularizers
    instance_offset_reg = spmat_from_hdf(hdf_file, string(path, "/instance_offset_reg"))
    feature_offset_reg = spmat_from_hdf(hdf_file, string(path, "/feature_offset_reg"))

    # Precisions
    feature_precision = hdf_file[string(path,"/feature_precision")][:]

    # Losses
    loss_names = hdf_file[string(path,"/loss_types")][:] 
    loss_scales = hdf_file[string(path,"/loss_scales")][:] 
    losses = Loss[eval(Meta.parse(lname))(lscale) for (lname, lscale) in zip(loss_names, loss_scales)]

    # Instance regularizers
    inst_reg_gp = hdf_file[string(path, "/inst_reg")]
    inst_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(inst_reg_gp))]
   
    # Feature regularizers
    feat_reg_gp = hdf_file[string(path, "/feat_reg")]
    feat_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(feat_reg_gp))]

    # Factor importances
    factor_importances = hdf_file[string(path,"/factor_importances")][:]

    return MatFacModel(X, Y, inst_reg_mats, feat_reg_mats,
                             instance_offset, instance_offset_reg, 
                             feature_offset, feature_offset_reg,
                             feature_precision,
                             covariate_coeff, covariate_coeff_reg,
                             losses, factor_importances) 
end


##################################
# SparseMatrixCSC
##################################

function to_hdf(hdf_file, path::String, mat::SparseMatrixCSC)

    write(hdf_file, string(path, "/colptr"), mat.colptr)
    write(hdf_file, string(path, "/m"), mat.m)
    write(hdf_file, string(path, "/n"), mat.n)
    write(hdf_file, string(path, "/nzval"), mat.nzval)
    write(hdf_file, string(path, "/rowval"), mat.rowval)

end

function spmat_from_hdf(hdf_file, path::String)
    colptr = read(hdf_file, string(path, "/colptr"))
    m = read(hdf_file, string(path, "/m"))
    n = read(hdf_file, string(path, "/n"))
    nzval = read(hdf_file, string(path, "/nzval"))
    rowval = read(hdf_file, string(path, "/rowval"))

    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end


