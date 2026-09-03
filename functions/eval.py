from sklearn.metrics import auc as auc_score
from scipy.stats import kendalltau
import numpy as np
# from sklearn.linear_model import LinearRegression
import kneeliverse.lmethod as lmethod
import kneeliverse.kneedle as kneedle
import kneeliverse.dfdt as dfdt
from .pred import class_proba, class_proba_batch

def hard_rationale_selection(
    token_weights,
    method='elbow',
    elbow_method='simple-lmethod',
    n=None,
    k=None
):
    # Sort by importance descending
    sorted_token_weights = sorted(
        token_weights,
        key=lambda x: x[2],
        reverse=True
    )

    # ORIGINAL token positions
    original_indices = [x[0] for x in sorted_token_weights]

    # Sorted importance scores
    scores = np.array([x[2] for x in sorted_token_weights])

    if method == 'elbow':

        if elbow_method is None:
            raise ValueError(
                "Elbow method requires a specific elbow_method"
            )

        if elbow_method == 'simple-lmethod':
            n = scores.size
            if n < 2:
                return original_indices[:n]

            x = np.arange(n, dtype=np.float64)
            x_mean = (n - 1) / 2.0
            y_mean = scores.mean()

            dx = x - x_mean
            slope = np.dot(dx, scores - y_mean) / np.dot(dx, dx)
            intercept = y_mean - slope * x_mean

            fitted = slope * x + intercept
            residuals = fitted - scores

            elbow_idx = np.argmax(residuals[1:]) + 1
            important_positions = original_indices[:elbow_idx]

        elif elbow_method == 'lmethod':
            try:
                rank_indices = np.arange(len(scores))
                points = np.column_stack(
                    (rank_indices, scores)
                )

                knee = lmethod.knee(points)

                knee_idx = int(
                    np.asarray(points[knee, 0]).flatten()[0]
                )

                important_positions = original_indices[:knee_idx]
            except Exception as e:
                important_positions = []

        elif elbow_method == 'kneedle':

            rank_indices = np.arange(len(scores))
            points = np.column_stack(
                (rank_indices, scores)
            )

            knee = kneedle.knee(points)

            knee_idx = int(
                np.asarray(points[knee, 0]).flatten()[0]
            )

            important_positions = original_indices[:knee_idx]

        elif elbow_method == 'dfdt':
            try:
                rank_indices = np.arange(len(scores))
                points = np.column_stack(
                    (rank_indices, scores)
                )

                knee = dfdt.knee(points)

                knee_idx = int(
                    np.asarray(points[knee, 0]).flatten()[0]
                )

                important_positions = original_indices[:knee_idx]
            except Exception as e:
                important_positions = []

        else:
            raise ValueError(
                f"Invalid elbow_method: {elbow_method}"
            )

    elif method == 'top_n':

        if n is None:
            raise ValueError(
                "Top-N method requires parameter n"
            )

        important_positions = original_indices[:n]

    elif method == 'threshold':

        if k is None:
            raise ValueError(
                "Threshold method requires parameter 'threshold'"
            )

        important_positions = [
            x[0]
            for x in sorted_token_weights
            if x[2] >= k
        ]

    else:
        raise ValueError(
            f"Invalid method: {method}"
        )

    return important_positions

def comprehensivness(tokenizer, pipe, predicted_class, xai_token_importance, proba_dict, prediction_cache, method='elbow', elbow_method='simple-lmethod', n=None, k=None):
    predicted_class_proba = proba_dict[predicted_class]

    # Automatically get the model/tokenizer mask token
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise ValueError(
            "The provided tokenizer does not define a mask token. "
            "This model may not support mask-based perturbation."
        )
    
    xai_token_importance2 = xai_token_importance.copy()

    tokens = [x[1] for x in xai_token_importance]
    important_positions = hard_rationale_selection(
        xai_token_importance2,
        method=method,
        elbow_method=elbow_method,
        n=n,
        k=k
    )

    # Replace important tokens with the model's mask token
    tokens_after_removal = [
        mask_token if i in important_positions else tok
        for i, tok in enumerate(tokens)
    ]

    comment_without_xai = tokenizer.convert_tokens_to_string(
        tokens_after_removal
    )

    if comment_without_xai not in prediction_cache:
        new_probability = class_proba(
            pipe,
            comment_without_xai
        )[predicted_class]
        prediction_cache[comment_without_xai] = new_probability
    else:
        new_probability = prediction_cache[comment_without_xai]

    return predicted_class_proba - new_probability



def sufficiency(tokenizer, pipe, predicted_class, xai_token_importance, proba_dict, prediction_cache, method='elbow', elbow_method='simple-lmethod', n=None, k=None):
    predicted_class_proba = proba_dict[predicted_class]

    # Automatically get the model/tokenizer mask token
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise ValueError(
            "The provided tokenizer does not define a mask token. "
            "This model may not support mask-based perturbation."
        )

    xai_token_importance2 = xai_token_importance.copy()

    tokens = [x[1] for x in xai_token_importance]
    important_positions = hard_rationale_selection(
        xai_token_importance2,
        method=method,
        elbow_method=elbow_method,
        n=n,
        k=k
    )

    # Keep important tokens and mask everything else
    tokens_with_xai_only = [
        tok if i in important_positions else mask_token
        for i, tok in enumerate(tokens)
    ]

    comment_with_xai_only = tokenizer.convert_tokens_to_string(
        tokens_with_xai_only
    )

    if comment_with_xai_only not in prediction_cache:
        new_probability = class_proba(
            pipe,
            comment_with_xai_only
        )[predicted_class]
        prediction_cache[comment_with_xai_only] = new_probability
    else:
        new_probability = prediction_cache[comment_with_xai_only]

    return predicted_class_proba - new_probability


def correlation_leave_one_out(
    tokenizer,
    pipe,
    predicted_class,
    xai_token_importance,
    proba_dict,
    prediction_cache,
    batch_size=256
):
    predicted_class_proba = proba_dict[predicted_class]
    mask_token = tokenizer.mask_token
    tokens = [x[1] for x in xai_token_importance]

    perturbed_texts = []
    for i in range(len(tokens)):
        masked_tokens = [
            mask_token if j == i else token
            for j, token in enumerate(tokens)
        ]
        perturbed_texts.append(
            tokenizer.convert_tokens_to_string(masked_tokens)
        )

    # Identify which texts need prediction vs which are cached
    texts_to_predict = []
    indices_to_predict = []
    
    for idx, text in enumerate(perturbed_texts):
        if text not in prediction_cache:
            texts_to_predict.append(text)
            indices_to_predict.append(idx)

    # Batch predict only the missing entries
    if texts_to_predict:
        batch_probabilities = class_proba_batch(
            pipe,
            texts_to_predict,
            batch_size=batch_size
        )
        # Update cache with new predictions
        for text, prob_dict in zip(texts_to_predict, batch_probabilities):
            prediction_cache[text] = prob_dict[predicted_class]

    # Retrieve probabilities from cache in original order
    loo_scores = [
        predicted_class_proba - prediction_cache[text]
        for text in perturbed_texts
    ]

    xai_scores = [x[2] for x in xai_token_importance]
    tau, _ = kendalltau(xai_scores, loo_scores)
    return tau


def insertion_auc(tokenizer, pipe, predicted_class, xai_token_importance, prediction_cache):
    """
    Optimized version using batch prediction for all perturbations of a single sample.
    """
    n = len(xai_token_importance)
    if n == 0:
        raise ValueError("xai_token_importance must contain at least one token.")
    
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise ValueError("Tokenizer does not define a mask token.")
    
    # Canonical original sequence
    tokens_by_position = sorted(
        [(x[0], x[1]) for x in xai_token_importance],
        key=lambda x: x[0]
    )
    
    # Rank token occurrences by attribution
    ranked_tokens = sorted(
        xai_token_importance,
        key=lambda x: x[2],
        reverse=True
    )
    
    revealed_positions = set()
    perturbed_texts = []
    
    # k = 0: all tokens masked
    masked_tokens = [mask_token for _position, _token in tokens_by_position]
    masked_comment = tokenizer.convert_tokens_to_string(masked_tokens)
    perturbed_texts.append(masked_comment)
    
    # k = 1, ..., n: progressively reveal tokens
    for position, token, weight in ranked_tokens:
        revealed_positions.add(position)
        inserted_tokens = [
            original_token
            if original_position in revealed_positions
            else mask_token
            for original_position, original_token in tokens_by_position
        ]
        inserted_comment = tokenizer.convert_tokens_to_string(inserted_tokens)
        perturbed_texts.append(inserted_comment)
    
    # Filter out texts already in cache
    texts_to_predict = [txt for txt in perturbed_texts if txt not in prediction_cache]
    
    if texts_to_predict:
        # Batch predict all new texts
        batch_probabilities = class_proba_batch(pipe, texts_to_predict)
        
        # Update cache
        for txt, prob_dict in zip(texts_to_predict, batch_probabilities):
            prediction_cache[txt] = prob_dict[predicted_class]
    
    # Retrieve probabilities from cache
    auc_scores = [prediction_cache[txt] for txt in perturbed_texts]
    
    # Normalized perturbation fractions
    x = [i / n for i in range(n + 1)]
    auc = auc_score(x, auc_scores)
    return auc

def deletion_auc(tokenizer, pipe, predicted_class, xai_token_importance, prediction_cache):
    """
    Optimized version using batch prediction for all perturbations of a single sample.
    
    Mask-based deletion curve:
        full input
             ->
         progressively replace tokens with the mask token
         in descending attribution order
             ->
         all masked
    AUC is computed over normalized perturbation fractions:
        0, 1/n, 2/n, ..., 1
    """
    n = len(xai_token_importance)
    if n == 0:
        raise ValueError("xai_token_importance must contain at least one token.")
    
    # Automatically get the model/tokenizer mask token
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise ValueError(
            "The provided tokenizer does not define a mask token. "
            "This model may not support mask-based perturbation."
        )
    
    # Canonical sequence in original positional order
    tokens_by_position = sorted(
        [(x[0], x[1]) for x in xai_token_importance],
        key=lambda x: x[0]
    )
    
    # Rank token occurrences by attribution
    ranked_tokens = sorted(
        xai_token_importance,
        key=lambda x: x[2],
        reverse=True
    )
    
    masked_positions = set()
    perturbed_texts = []
    
    # k = 0: original full-input prediction
    full_tokens = [
        token
        for position, token in tokens_by_position
    ]
    full_comment = tokenizer.convert_tokens_to_string(full_tokens)
    perturbed_texts.append(full_comment)
    
    # k = 1, ..., n:
    # progressively replace ranked token occurrences with mask
    for position, token, weight in ranked_tokens:
        masked_positions.add(position)
        remaining_tokens = [
            mask_token
            if original_position in masked_positions
            else original_token
            for original_position, original_token in tokens_by_position
        ]
        remaining_comment = tokenizer.convert_tokens_to_string(remaining_tokens)
        perturbed_texts.append(remaining_comment)
    
    # Filter out texts already in cache
    texts_to_predict = [txt for txt in perturbed_texts if txt not in prediction_cache]
    
    if texts_to_predict:
        # Batch predict all new texts
        # Note: You might need to import class_proba_batch or pass it as an argument
        from .pred import class_proba_batch
        batch_probabilities = class_proba_batch(pipe, texts_to_predict)
        
        # Update cache
        for txt, prob_dict in zip(texts_to_predict, batch_probabilities):
            prediction_cache[txt] = prob_dict[predicted_class]
    
    # Retrieve probabilities from cache
    auc_scores = [prediction_cache[txt] for txt in perturbed_texts]
    
    # Normalized perturbation fractions
    x = [i / n for i in range(n + 1)]
    auc = auc_score(x, auc_scores)
    return auc

def combined_metric(comp, suff, corr_loo, ins_auc, del_auc):
    comb = 0
    nb_metrics = 0
    if comp is not None:
        comb += comp
        nb_metrics += 1
    if suff is not None:
        comb += (1 - suff)
        nb_metrics += 1
    if corr_loo is not None:
        comb += ((corr_loo + 1) / 2)
        nb_metrics += 1
    if ins_auc is not None:
        comb += ins_auc
        nb_metrics += 1
    if del_auc is not None:
        comb += (1 - del_auc)
        nb_metrics += 1
    if nb_metrics == 0:
        return None
    return comb / nb_metrics

def borda_count(eval_df):
    """
    takes as input a dataframe with:
    - columns: "comprehensiveness", "sufficiency", "corr_loo", "ins_AUC", "del_AUC"
    - row indices: different XAI methods names
    - cell values: the corresponding metric scores for each method
    returns the dataframe with the Borda count score column added for each method
    """

    higher_is_better = {
        "comprehensiveness": True, "sufficiency": False, "corr_loo": True, 
        "ins_AUC": True, "del_AUC": False
    }

    for metric, hib in higher_is_better.items():
        if hib:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=False, method='min')
        else:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=True, method='min')

    # Borda count score is the sum of ranks across all metrics
    rank_columns = [metric + "_rank" for metric in higher_is_better.keys()]
    eval_df["borda"] = eval_df[rank_columns].sum(axis=1).astype(int)
    eval_df.drop(columns=rank_columns, inplace=True)
    return eval_df

def borda_count_leave_one_out(eval_df, metric_to_leave_out=None):
    """
    takes as input a dataframe with:
    - columns: "comprehensiveness", "sufficiency", "corr_loo", "ins_AUC", "del_AUC" with one metric optionally left out
    - row indices: different XAI methods names
    - cell values: the corresponding metric scores for each method
    returns the dataframe with the Borda count score column added for each method
    """
    higher_is_better = {
        "comprehensiveness": True, "sufficiency": False, "corr_loo": True, 
        "ins_AUC": True, "del_AUC": False
    }

    if metric_to_leave_out is not None:
        del higher_is_better[metric_to_leave_out]

    for metric, hib in higher_is_better.items():
        if hib:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=False, method='min')
        else:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=True, method='min')

    # Borda count score is the sum of ranks across all metrics
    rank_columns = [metric + "_rank" for metric in higher_is_better.keys()]
    eval_df["borda"] = eval_df[rank_columns].sum(axis=1).astype(int)
    eval_df.drop(columns=rank_columns, inplace=True)
    return eval_df

def borda_count_hard_rationale(eval_df):
    """
    takes as input a dataframe with:
    - columns: "comprehensiveness", "sufficiency"
    - row indices: different XAI methods names
    - cell values: the corresponding metric scores for each method
    returns the dataframe with the Borda count score column added for each method
    """

    higher_is_better = {
        "comprehensiveness": True, "sufficiency": False
    }

    for metric, hib in higher_is_better.items():
        if hib:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=False, method='min')
        else:
            eval_df[metric + "_rank"] = eval_df[metric].rank(ascending=True, method='min')

    # Borda count score is the sum of ranks across all metrics
    rank_columns = [metric + "_rank" for metric in higher_is_better.keys()]
    eval_df["borda"] = eval_df[rank_columns].sum(axis=1).astype(int)
    eval_df.drop(columns=rank_columns, inplace=True)
    return eval_df