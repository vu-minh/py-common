# import os
# import joblib
import random
import numpy as np


SIM_LINK_TYPE = 1
DIS_LINK_TYPE = -1


def gen_similar_links(labels, n_links, include_link_type=True, seed=42):
    """Generate similar link (Must-link) constraints.
    """
    # Reference: https://realpython.com/python-random/

    min_class_id, max_class_id = labels.min(), labels.max()
    n_gen = 0
    links = []

    random.seed(seed)
    while n_gen < n_links:
        # pick random a class
        # random.randint(x, y) -> its range is [x, y]
        # random.randrange(x, y) -> its range is [x, y)
        c = random.randint(min_class_id, max_class_id)

        # filter the indices of points in this class
        (point_idx,) = np.where(labels == c)

        # pick random two indices in this class
        # do not use random.choices(items, k)
        # -> do with replacement -> duplicates are possible
        p1, p2 = random.sample(point_idx.tolist(), 2)

        # store the sampled indices with `link_type`=1
        links.append([p1, p2, SIM_LINK_TYPE] if include_link_type else [p1, p2])
        n_gen += 1

    return np.array(links)


def gen_dissimilar_links(labels, n_links, include_link_type=True, seed=42):
    """Generate dissimilar link (cannot-link) constraints.
    """
    min_class_id, max_class_id = labels.min(), labels.max()
    n_gen = 0
    links = []

    random.seed(seed)
    while n_gen < n_links:
        # pick 2 random different classes
        # not to use random.sample(items, k) to ensure random WITHOUT replacement
        c1, c2 = random.sample(range(int(min_class_id), int(max_class_id) + 1), 2)

        # filter point indices for each selected class, and take a sample for each class
        (idx1,) = np.where(labels == c1)
        (idx2,) = np.where(labels == c2)

        p1 = random.choices(idx1.tolist())[0]
        p2 = random.choices(idx2.tolist())[0]

        # store the generated link with `link_type`=-1
        links.append([p1, p2, DIS_LINK_TYPE] if include_link_type else [p1, p2])
        n_gen += 1

    return np.array(links)


def generate_contrastive_constraints(labels, n_links=10, seed=42):
    """The contrastive constraints, in a narrow meaning, can be considered as a triplet(x, x+, x-),
        in which, x is a sample, x+ is a positive sample (similar to x)
        and x- is a negative sample (dissimilar to x)

        Can be used to construct a constraint-preserving score as follow:
            - log ( exp(f(x)T f(x+)) / (exp(f(x)Tf(x+)) + exp(f(x)Tf(x-)))  )

        [1] S. Arora, H. Khandeparkar, M. Khodak, O. Plevrakis, and N. Saunshi,
        “A Theoretical Analysis of Contrastive Unsupervised Representation Learning,” 2019.
    """
    min_class_id, max_class_id = labels.min(), labels.max()
    n_gen = 0
    links = []

    random.seed(seed)
    while n_gen < n_links:
        # pick 2 random different classes
        # not to use random.sample(items, k) to ensure random WITHOUT replacement
        c1, c2 = random.sample(range(int(min_class_id), int(max_class_id) + 1), 2)

        # filter point indices for each selected class
        (idx1,) = np.where(labels == c1)
        (idx2,) = np.where(labels == c2)

        # random sample 2 points in the same class
        px, px_positive = random.sample(idx1.tolist(), k=2)

        # random the third negative sample from the other class
        px_negative = random.choices(idx2.tolist())[0]

        links.append([px, px_positive, px_negative])
        n_gen += 1

    return links
