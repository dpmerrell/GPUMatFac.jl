
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

end


function matfac_from_hdf(hdf_file, path::String)
    # Factors
    X = hdf_file[string(path,"/X")][:,:]
    Y = hdf_file[string(path,"/Y")][:,:]

    # Losses
    loss_names = hdf_file[string(path,"/loss_types")][:] 
    loss_scales = hdf_file[string(path,"/loss_scales")][:] 
    losses = Loss[loss_map[lname](lscale) for (lname, lscale) in zip(loss_names, loss_scales)]

    # Instance regularizers
    inst_reg_gp = hdf_file[string(path, "/inst_reg")]
    inst_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(inst_reg_gp))]
   
    # Feature regularizers
    feat_reg_gp = hdf_file[string(path, "/feat_reg")]
    feat_reg_mats = SparseMatrixCSC[spmat_from_hdf(inst_reg_gp, k) for k in sort(keys(feat_reg_gp))]

    return MatFacModel(X, Y, losses, inst_reg_mats, feat_reg_mats) 
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


