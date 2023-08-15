import fire
import scipy
import kaldiio
import torch
import numpy as np
from tqdm import tqdm


def key_to_class(sample_key):
    """
    Specific to VoxCeleb1
    """
    return sample_key[:7]

def load_class_indices(embed_scp):
    classes = set()
    with open(embed_scp, 'r') as f:
        for line in f:
            key, embed_path = line.split(maxsplit=1)
            classes.add(key_to_class(key))
    class2id = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
    return class2id

def load_weights(checkpoint_path, num_classes):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    weights = state_dict["projection.weight"]
    assert weights.shape[0] % num_classes == 0
    weights = weights[:num_classes, :]
    return weights.cpu().numpy()

def compute_nc(embed_scp, moving_average=False, classifier_weights=None, checkpoint_path=None, **kwargs):
    class2id = load_class_indices(embed_scp)
    num_classes = len(class2id)
    print("Number of speakers: %d" % (num_classes))

    # Iteration 1 -> between-class statistics
    N_class = torch.zeros(num_classes)
    N = 0
    for key, embed in tqdm(kaldiio.load_scp_sequential(embed_scp)):
        class_idx = class2id[key_to_class(key)]
        embed = torch.from_numpy(embed)

        N_class_batch = torch.zeros(num_classes)
        N_class_batch[class_idx] = 1

        hG_batch_sum = embed
        hbar_batch_sum = torch.zeros(embed.size(0), num_classes)
        hbar_batch_sum[:, class_idx] = embed

        if N == 0:
            hG = hG_batch_sum
            hbar = hbar_batch_sum
        elif moving_average:
            hG = (N / (N + 1)) * hG + (1 / (N + 1)) * hG_batch_sum
            hbar = (N_class / torch.clamp(N_class + N_class_batch, 1)) * hbar + \
                (1 / torch.clamp(N_class + N_class_batch, 1)) * hbar_batch_sum
        else:
            hG += hG_batch_sum
            hbar += hbar_batch_sum

        N += 1
        N_class += N_class_batch

    if not moving_average:
        hG = hG / N
        hbar = hbar / torch.clamp(N_class, 1)

    # Iteration 2 -> within-class statistics
    N = 0
    for key, embed in tqdm(kaldiio.load_scp_sequential(embed_scp)):
        class_idx = class2id[key_to_class(key)]
        embed = torch.from_numpy(embed)

        h_mean_diff = (embed - hbar[:, class_idx]).unsqueeze(1)
        Sigma_W_batch_sum = (h_mean_diff @ h_mean_diff.T)

        if N == 0:
            Sigma_W = Sigma_W_batch_sum
        elif moving_average:
            Sigma_W = (N / (N + 1)) * Sigma_W + (1 / (N + 1)) * Sigma_W_batch_sum
        else:
            Sigma_W += Sigma_W_batch_sum

        N += 1

    if not moving_average:
        Sigma_W = Sigma_W / N

    # Between-class covariance
    hbar_mean_diff = hbar - hG.unsqueeze(1)
    Sigma_B = (hbar_mean_diff @ hbar_mean_diff.T) / num_classes

    K = num_classes
    Sigma_W = Sigma_W.cpu().numpy()
    Sigma_B = Sigma_B.cpu().numpy()

    # NC1
    nc1 = 1 / K * np.trace(Sigma_W @ scipy.linalg.pinv(Sigma_B))
    print("NC1: %.10f" % (nc1))

    W = None
    if checkpoint_path:
        W = load_weights(checkpoint_path, num_classes).T
    elif classifier_weights:
        W = np.load(classifier_weights).T
    
    if W is not None:
        assert W.shape[1] == num_classes, f"Class numbers mismatch! ({W.size(1), num_classes})"
        assert W.shape[0] == hbar.size(0), f"Feature dimensions mismatch! ({W.size(0), hbar.size(0)})"
        H_bar = hbar.cpu().numpy()

        # NC2
        gram = W.T @ W
        nc2 = np.linalg.norm(gram / np.linalg.norm(gram) - np.sqrt(1 / (K-1))*(np.eye(K) - 1/K))
        # nc2 = np.linalg.norm(W.T @ W - (K / (K-1) * (np.eye(K) - (1 / K))))

        # NC3
        dual = W.T @ H_bar
        nc3 = np.linalg.norm(dual / np.linalg.norm(dual) - np.sqrt(1 / (K-1))*(np.eye(K) - 1/K))
        # nc3 = np.linalg.norm(W.T @ H_bar - (K / (K-1) * (np.eye(K) - (1 / K))))
        print("NC2: %.5f" % (nc2))
        print("NC3: %.5f" % (nc3))


if __name__ == "__main__":
    fire.Fire(compute_nc)
