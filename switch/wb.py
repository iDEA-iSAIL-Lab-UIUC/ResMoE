from dis import dis
import torch
import ot
import time, pickle
import n_sphere

def is_diagonal(w, eps=1e-6):
    w_abs = torch.abs(w)
    return abs(torch.diagonal(w_abs).sum().item() - w_abs.sum().item()) < eps

def get_optimal_permutation(w, measures_weights, wc, b=None, numThreads="max"):
    """
    compute the mean of \|T_i W_i - W_c\|_F^2

    b: mass for points in wc
    weights: 
    """
    nx = ot.backend.get_backend(w)
    
    # print(nx,wc)

    k = wc.shape[0]
    if b is None:
        b = nx.ones((k,), type_as=wc) / k
        
    # print(b)

    # torch.long to make sure it can be taken as index in PyTorch
    res = torch.zeros((w.shape[0], k), dtype=torch.long)

    i = 0
    for (measure_locations_i, measure_weights_i) in zip(w, measures_weights):
        M_i = ot.utils.dist(wc, measure_locations_i)
        # print(M_i.shape, measure_locations_i.shape, 
        #   measure_weights_i.shape, b.shape)
        T_i = ot.lp.emd(b, measure_weights_i, M_i, numThreads=numThreads)

        # T_i has the same shape as M_i, so T_i is from wc to w
        
        
        print(is_diagonal(T_i))

        res[i] = T_i.argmax(1)
        i += 1
    
    return res


def get_expert_weights(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    w_dict = state_dict['model']

    l = state_dict['args'].num_layers
    n = state_dict['args'].num_experts[0]
    e_layers = [i for i in range(l) 
                        if (i+1)%state_dict['args'].expert_interval == 0]

    suffix = "encoderlayers.{}.mlp.deepspeed_moe.experts.deepspeed_experts.{}."

    ini_l = e_layers[0]
    shape_w1 = w_dict[suffix.format(ini_l, 0) + "dense_h_to_4h.weight"].shape
    shape_b1 = w_dict[suffix.format(ini_l, 0) + "dense_h_to_4h.bias"].shape
    shape_w2 = w_dict[suffix.format(ini_l, 0) + "dense_4h_to_h.weight"].shape
    shape_b2 = w_dict[suffix.format(ini_l, 0) + "dense_4h_to_h.bias"].shape

    w1 = torch.zeros((len(e_layers), n) + shape_w1)  # [1, n, 2048, 512]
    b1 = torch.zeros((len(e_layers), n) + shape_b1)
    w2 = torch.zeros((len(e_layers), n) + shape_w2)
    b2 = torch.zeros((len(e_layers), n) + shape_b2)

    measures_weights = torch.ones(b1.shape) / shape_b1[0]

    for i in range(len(e_layers)):
        for j in range(n):
            w1[i, j] = w_dict[suffix.format(e_layers[i], j) 
                                + "dense_h_to_4h.weight"]
            b1[i, j] = w_dict[suffix.format(e_layers[i], j) 
                                + "dense_h_to_4h.bias"]
            w2[i, j] = w_dict[suffix.format(e_layers[i], j) 
                                + "dense_4h_to_h.weight"]
            b2[i, j] = w_dict[suffix.format(e_layers[i], j) 
                                + "dense_4h_to_h.bias"]
    
    return (w1, w2, b1, b2, measures_weights)

def get_approx_loss(w1, w2, wc1, wc2, T=None, factor=None):

    def avg_loss(w, wc=None, factor=None):
        if factor is None: factor = w.shape[0]
        if wc is None:
            return (w * w).sum().item() / factor
        else:
            # print(w.shape,wc.shape)
            diff = w - wc
            return (diff * diff).sum().item() / factor

    n_e = w1.shape[0] if factor is None else factor

    # ref = avg_loss(w1) + avg_loss(w2)
    # print('reference scale', ref)
    
    if T is None:
        error = avg_loss(w1, wc1, n_e) + avg_loss(w2, wc2, n_e)
        # print('approx error', error)
    
    else:
        error = 0
        for i in range(T.shape[0]):
            # error += avg_loss(w1[i], wc1[T[i]]) + avg_loss(w2[i], wc2[:, T[i]])
            # error += avg_loss(w1[i][T[i]], wc1, n_e) + avg_loss(w2[i], wc2, n_e)
            if len(wc1.shape) == 3:
                error += avg_loss(w1[i][T[i]], wc1[i], n_e) + avg_loss(w2[i][:, T[i]], wc2[i], n_e)
            else:
                error += avg_loss(w1[i][T[i]], wc1, n_e) + avg_loss(w2[i][:, T[i]], wc2, n_e)

        # print('approx error', error)
    
    return error

def get_LoRA(tgt, r=32):

    lora0 = (tgt[0].Vh[:,:,:r], tgt[0].U[:,:,:,:r] * tgt[0].S[:,:,None,:r])
    lora1 = (tgt[1].Vh[:,:,:r], tgt[1].U[:,:,:,:r] * tgt[1].S[:,:,None,:r])
    
    return (lora0, lora1)

def permute_weights(w1, w2, T):
    if len(T.shape) == 3:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                w1[i][j] = w1[i][j][ T[i][j] ]
                w2[i][j] = w2[i][j][ :, T[i][j] ]
    elif len(T.shape) == 2:
        for j in range(T.shape[0]):
            w1[j] = w1[j][ T[j] ]
            w2[j] = w2[j][ :, T[j] ]


def weights_permute(w1, w2, T):
    permuted_w1 = torch.empty_like(w1)
    permuted_w2 = torch.empty_like(w2)

    # Permute each batch and concatenate
    if len(T.shape)==2:
      for i in range(T.size(0)):
          permuted_w1[i] = w1[i, T[i]]
          permuted_w2[i] = w2[i, :, T[i]]
          
    elif len(T.shape)==1:
      for i in range(T.size(0)):
          permuted_w1[i] = w1[T[i],:]
          permuted_w2[:,i] = w2[:, T[i]]
        
    return permuted_w1, permuted_w2
    
    
    # if len(T.shape) == 3:
    #     for i in range(T.shape[0]):
    #         for j in range(T.shape[1]):
    #           wr1[i][j] = w1[i][ T[i][j] ]
    #           wr2[i][j] = w2[i][ :, T[i][j] ]
    # elif len(T.shape) == 2:
    #     for j in range(T.shape[0]):
    #         w1[j] = w1[j][ T[j] ]
    #         w2[j] = w2[j][ :, T[j] ]


def cart2sphere(w):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            print(w[i][j])
            w[i][j] = n_sphere.convert_spherical(w[i][j])

def sphere2cart(w):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] = n_sphere.convert_rectangular(w[i][j])

def extract_mean_dist(w1, w2, measures_weights):
    # w1: e x 4h x h
    w_cat = torch.cat((w1, w2.transpose(1, 2)), dim=2)

    res = ot.lp.free_support_barycenter(w_cat, measures_weights, 
            X_init=w_cat.mean(0), numThreads="max")

    T = get_optimal_permutation(w_cat, measures_weights, res)

    res1 = res[:, :h]
    res2 = res[:, h:].transpose(0, 1) # 1024 x 256 -> 256 x 1024

    get_approx_loss(w1, w2, res1, res2, T)

    return (res1, res2, T)




if __name__ == "__main__":
    
    checkpoint_name = "pretrain_gpts_moe32k1"
    # directory = "iter_0010000"
    directory = "release"

    checkpoint = "./checkpoints/{}/{}/mp_rank_00/model_optim_rng.pt".format(
        checkpoint_name, directory)

    w1, w2, b1, b2, measures_weights = get_expert_weights(checkpoint)
    h = w1.shape[-1]
    n_layer = w1.shape[0]
    n_expert = w1.shape[1]

    # print(b2) # b2 in girish0 is all-zero

    # device = "cuda:0"
    # displacement = 1

    ### baseline: only find the barycenter of the first layer
    '''
    # wc1 = w1[0].mean(0)
    # wc2 = w2[0].mean(0)           
    wc1 = w1.mean(1)
    bc1 = b1.mean(1)
    wc2 = w2.mean(1)
    bc2 = b2.mean(1)

    print(wc1.shape, bc2.shape)

    with open(r"barycenter/pretrain_gpts_moe32k1_avg.pickle", "wb") as output_file:
        pickle.dump(((wc1, wc2), (bc1, bc2)), output_file)

    # get_approx_loss(w1[0], w2[0], wc1, wc2) # approx error 5137.0972900390625
    '''
    ### baseline: only find the barycenter of the first layer
    '''
    """
    res = ot.lp.free_support_barycenter(w1[0], measures_weights[0], 
            X_init=w1[0].mean(0), numThreads="max")

    T = get_optimal_permutation(w1[0], measures_weights[0], res)
    with open(r"pretrain_gpts_moe32k1_w1only_wb.pickle", "wb") as output_file:
        pickle.dump((res, T), output_file)
    """
    with open("pretrain_gpts_moe32k1_w1only_wb.pickle", 'rb') as handle: 
        (res, T) = pickle.load(handle)

    res2 = torch.zeros_like(w2[0].mean(0))
    m = T.shape[0]
    for i in range(m):
        res2 = res2 + w2[0][i][:, T[i]] / m

    # get_approx_loss(w1[0], w2[0], res, wc2)
    get_approx_loss(w1[0], w2[0], res, res2, T) # approx error 5163.94833946228
    '''

    ### proposed: find the barycenter of the combined representation
    
    # res: l x 4h x 2h
    # res1: l x 4h x h, res2: l x h x 4h
    # T: l x e x 4h
    """
    res1 = torch.zeros_like(w1.mean(1))
    res2 = torch.zeros_like(w2.mean(1))
    T = torch.zeros((n_layer, w1.shape[1], w1.shape[2]), dtype=torch.long)

    if displacement:
        # for test
        n_layer = 1
        w1 = w1 - w1.mean(1, keepdim=True)
        w2 = w2 - w2.mean(2, keepdim=True)

    for i in range(n_layer):
        # if displacement:
            # expert_displacement = w_cat.mean(1, keepdim=True)
            # w_cat = w_cat - expert_displacement

        w_cat = torch.cat((w1[i], w2[i].transpose(1, 2)), dim=2)

        res = ot.lp.free_support_barycenter(w_cat, measures_weights[i], 
                X_init=w_cat.mean(0), numThreads="max")

        T[i] = get_optimal_permutation(w_cat, measures_weights[i], res)

        res1[i] = res[:, :h]
        res2[i] = res[:, h:].transpose(0, 1) # 1024 x 256 -> 256 x 1024

        # if displacement:
        #     res1 = res1 + expert_displacement[:,:,:h]
        #     res2 = res2 + expert_displacement[:,:,h:].transpose(1, 2)

        get_approx_loss(w1[i], w2[i], res1[i], res2[i], T[i])

    
    # with open(r"barycenter/{}_wb.pickle".format(checkpoint_name), "wb") as output_file: 
    with open(r"barycenter/{}_wb_dis.pickle".format(checkpoint_name), "wb") as output_file: 
        pickle.dump((res1, res2, T), output_file)
    
    # with open("barycenter/{}_wb.pickle".format(checkpoint_name), 'rb') as handle: 
    with open("barycenter/{}_wb_dis.pickle".format(checkpoint_name), 'rb') as handle: 
        res1, res2, T = pickle.load(handle)
        # _, _, (res1, res2), T = pickle.load(handle)
        # (lora1, lora2), (b1, b2), (res1, res2), T = pickle.load(handle)
    get_approx_loss(w1[0], w2[0], res1[0], res2[0], T[0])
    
    # res1 = w1.mean(1)
    # res2 = w2.mean(1)
    
    permute_weights(w1, w2, T)
    # r = 32
    """


    '''
    w10 = w1.clone()
    w20 = w2.clone()

    cart2sphere(w1)
    cart2sphere(w2)

    res1 = w1.mean(1)
    res2 = w2.mean(1)

    
    tgt1 = w1 - res1[:, None, :, :]
    tgt1 = torch.linalg.svd(tgt1, full_matrices=False) # leph
    # lora1 = (tgt1.Vh[:,:,:r], tgt1.U[:,:,:,:r] * tgt1.S[:,:,None,:r], )

    tgt2 = w2 - res2[:, None, :, :]
    tgt2 = torch.linalg.svd(tgt2, full_matrices=False) # leph
    # lora2 = (tgt2.Vh[:,:,:r], tgt2.U[:,:,:,:r] * tgt2.S[:,:,None,:r], )
    
    lora1 = res1[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt1.U[:,:,:,:r], tgt1.S[:,:,:r], tgt1.Vh[:,:,:r])
    lora2 = res2[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt2.U[:,:,:,:r], tgt2.S[:,:,:r], tgt2.Vh[:,:,:r])

    
    tgt1 = w1 - res1[:, None, :, :]
    tgt1 = torch.linalg.svd(tgt1, full_matrices=False) # leph
    # tgt1 = res1[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
    #         tgt1.U[:,:,:,:r], tgt1.S[:,:,:r], tgt1.Vh[:,:,:r])

    tgt2 = w2 - res2[:, None, :, :]
    tgt2 = torch.linalg.svd(tgt2, full_matrices=False) # leph
    # tgt2 = res2[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
    #         tgt2.U[:,:,:,:r], tgt2.S[:,:,:r], tgt2.Vh[:,:,:r])
    '''
    
    """
    # no better than simple A-M
    tgt1 = torch.exp(w1 - res1[:, None, :, :])
    tgt1 = torch.linalg.svd(tgt1, full_matrices=False) # leph
    tgt1 = torch.exp(res1[:, None, :, :]) * torch.einsum("...ph,...h,...hd->...pd",
        tgt1.U[:,:,:,:r], tgt1.S[:,:,:r], tgt1.Vh[:,:,:r]).abs()
    lora1 = torch.log(tgt1)

    tgt2 = torch.exp(w2 - res2[:, None, :, :])
    tgt2 = torch.linalg.svd(tgt2, full_matrices=False) # leph
    tgt2 = torch.exp(res2[:, None, :, :]) * torch.einsum("...ph,...h,...hd->...pd",
        tgt2.U[:,:,:,:r], tgt2.S[:,:,:r], tgt2.Vh[:,:,:r]).abs()
    lora2 = torch.log(tgt2)  
    

    sphere2cart(lora1)
    sphere2cart(lora2)


    get_approx_loss(w10[0], w20[0], lora1[0], lora2[0])
    # get_approx_loss(w1[0], w2[0], lora1[0], lora2[0])
    # get_approx_loss(w1[0], w2[0], tgt1[0], tgt2[0], T[0])
    

    mode = "wb_dis" if displacement else "wb"
    with open(r"barycenter/{}_{}.pickle".format(checkpoint_name, mode), 
                "wb") as output_file: 
        pickle.dump(((tgt1, tgt2), (b1, b2), (res1, res2), T), output_file)
    """
    ### proposed: first svd then barycenter then svd
    """
    res1 = torch.zeros_like(w1.mean(1))
    res2 = torch.zeros_like(w2.mean(1))
    T = torch.zeros((n_layer, w1.shape[1], w1.shape[2]), dtype=torch.long)

    r = 16

    tgt1 = torch.linalg.svd(w1, full_matrices=False)
    tgt2 = torch.linalg.svd(w2, full_matrices=False)

    lora1 = torch.einsum("...ph,...h,...hd->...pd",
            tgt1.U[:,:,:,:r], tgt1.S[:,:,:r], tgt1.Vh[:,:,:r])
    lora2 = torch.einsum("...ph,...h,...hd->...pd",
            tgt2.U[:,:,:,:r], tgt2.S[:,:,:r], tgt2.Vh[:,:,:r])
    get_approx_loss(w1[0], w2[0], lora1[0], lora2[0])

    w1 = w1 - lora1
    w2 = w2 - lora2

    for i in range(1):
        res1[i], res2[i], T[i] = extract_mean_dist(w1[i], w2[i], measures_weights[i])
    
    
    permute_weights(w1, w2, T[i][None,:])

    tgt1 = w1 - res1[:, None, :, :]
    tgt1 = torch.linalg.svd(tgt1, full_matrices=False) # leph

    tgt2 = w2 - res2[:, None, :, :]
    tgt2 = torch.linalg.svd(tgt2, full_matrices=False) # leph

    lora1 = res1[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt1.U[:,:,:,:r], tgt1.S[:,:,:r], tgt1.Vh[:,:,:r])
    lora2 = res2[:, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt2.U[:,:,:,:r], tgt2.S[:,:,:r], tgt2.Vh[:,:,:r])
    
    get_approx_loss(w1[0], w2[0], lora1[0], lora2[0])
    """
    
    ### proposed: clustering -> multiple barycenters
    with open("barycenter/{}_wb.pickle".format(checkpoint_name), 'rb') as handle: 
        (lora1, lora2), (b1, b2), (res1, res2), T = pickle.load(handle)

    w1 = w1[:1]; 
    w2 = w2[:1]; 
    res1 = res1[:1]; res2 = res2[:1]; T = T[:1]
    permute_weights(w1, w2, T)

    from sklearn.cluster import KMeans

    # X = torch.cat((w1, w2.transpose(-2, -1)), dim=2).reshape((1, n_expert, -1))[0]
    w_cat = torch.cat((w1, w2.transpose(-2, -1)), dim=2)
    X = w_cat[0].mean(1)

    
    k = 2
    r = (11-k) * 32 // 5

    labels_ = KMeans(n_clusters=k, n_init=30).fit_predict(X.detach().numpy())
    # kmeans = KMeans(n_clusters=3).fit_predict(X.detach().numpy())


    res1 = torch.zeros(1, k, 4*h, h)
    res2 = torch.zeros(1, k, h, 4*h)
    T = torch.zeros((n_layer, n_expert, 4*h), dtype=torch.long)

    for i in range(k):
        clus_idx = (labels_==i)
        n_i = clus_idx.sum()
        print(n_i)

        w1_clus = w1[0][clus_idx]
        w2_clus = w2[0][clus_idx]
        wts_clus = measures_weights[0][clus_idx]
        res1[0, i], res2[0, i], T[0, clus_idx] = extract_mean_dist(
            w1_clus, w2_clus, wts_clus)

        permute_weights(w1_clus, w2_clus, T[0, clus_idx])

        tgt1 = w1_clus - res1[:, i][0, None, :, :]
        tgt1 = torch.linalg.svd(tgt1, full_matrices=False) # leph

        tgt2 = w2_clus - res2[:, i][0, None, :, :]
        tgt2 = torch.linalg.svd(tgt2, full_matrices=False) # leph

        lora1 = res1[:, i][0, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt1.U[:,:,:r], tgt1.S[:,:r], tgt1.Vh[:,:r])
        lora2 = res2[:, i][0, None, :, :] + torch.einsum("...ph,...h,...hd->...pd",
            tgt2.U[:,:,:r], tgt2.S[:,:r], tgt2.Vh[:,:r])
        
        get_approx_loss(w1_clus, w2_clus, lora1, lora2, factor=32)




    ################################################################################
    '''
    T = get_optimal_permutation(w1[0, :5], measures_weights[0, :5], w1[0].mean(0))


    # w1=w1.to(device); measure_weights=measure_weights.to(device)
    s = time.time()
    res = ot.lp.free_support_barycenter(w1[0], measures_weights[0], 
            X_init=w1[0].mean(0), numThreads="max")
    print(time.time()-s)



    import pickle

    a = {'hello': 'world'}

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)
    '''

    """
    for i in range(n_layer):
        # if displacement:
            # expert_displacement = w_cat.mean(1, keepdim=True)
            # w_cat = w_cat - expert_displacement

        w_cat = torch.cat((w1[i], w2[i].transpose(1, 2)), dim=2)

        res = ot.lp.free_support_barycenter(w_cat, measures_weights[i], 
                X_init=w_cat.mean(0), numThreads="max")

        T[i] = get_optimal_permutation(w_cat, measures_weights[i], res)

        res1[i] = res[:, :h]
        res2[i] = res[:, h:].transpose(0, 1) # 1024 x 256 -> 256 x 1024

        # if displacement:
        #     res1 = res1 + expert_displacement[:,:,:h]
        #     res2 = res2 + expert_displacement[:,:,h:].transpose(1, 2)

        get_approx_loss(w1[i], w2[i], res1[i], res2[i], T[i])
    """