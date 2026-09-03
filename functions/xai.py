from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from lime import lime_text
import shap
from captum.attr import IntegratedGradients as CaptumIG, DeepLift
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functions.pred import remove_chaklas

class LimeExplainer:
    def __init__(self, model_name, device, num_samples=100, random_state=0):
        self.num_samples = num_samples
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        self.random_state = random_state
        self.explainer = lime_text.LimeTextExplainer(class_names=list(self.model.config.label2id.keys()), 
                            split_expression=self.tokenize, feature_selection='none', random_state=self.random_state)

    def predictor(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        probas = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        return probas

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                tokens[i] = tokens[i][2:]
        return tokens

    def explain(self, text, label, normalize=True):
        label_id = self.model.config.label2id[label]
        text = remove_chaklas(text)
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        reconstructed_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
        
        # these two lines are used to make sure random state works
        self.explainer.random_state = np.random.RandomState(self.random_state)
        self.explainer.base.random_state = self.explainer.random_state
        
        exp = self.explainer.explain_instance(reconstructed_text, self.predictor, num_samples=self.num_samples, labels=[label_id])
        lime_exp = exp.as_list(self.model.config.label2id[label])
        lime_exp_dict = dict(lime_exp)

        if normalize:
            min_score = min(lime_exp_dict.values())
            max_score = max(lime_exp_dict.values())
            if max_score != min_score:
                for token in lime_exp_dict.keys():
                    lime_exp_dict[token] = 2 * (lime_exp_dict[token] - min_score) / (max_score - min_score) - 1
            else:
                for token in lime_exp_dict.keys():
                    lime_exp_dict[token] = 0.0
        
        lime_exp = [[i, token, lime_exp_dict.get(token, lime_exp_dict.get(token[2:], 0))] for i, token in enumerate(tokens)]
        return lime_exp
    
class ShapExplainer:
    def __init__(self, model_name, device, max_evals=100, random_state=0):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        polarities = list(self.model.config.id2label.values())
        
        self.explainer = shap.Explainer(self.predict, self.tokenizer, algorithm="partition", 
                                output_names=polarities, max_evals=max_evals, seed=random_state)

    def predict(self, texts):
        texts = list(texts)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,   # good for Tensor Cores
        ).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return probs.float().cpu().numpy()

    def explain(self, text, label, normalize=True):
        text = remove_chaklas(text)
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        shap_values = self.explainer([text])
        values = shap_values.values[0].tolist()[1:-1]
        polarity_idx = self.model.config.label2id[label]

        # Extract SHAP value only for requested class
        scores = [value[polarity_idx] for value in values]

        if normalize:
            min_score = min(values, key=lambda x: x[polarity_idx])[polarity_idx]
            max_score = max(values, key=lambda x: x[polarity_idx])[polarity_idx]
            if max_score != min_score:
                scores = [
                    2 * (score - min_score) / (max_score - min_score) - 1
                    for score in scores
                ]
            else:
                scores = [0.0 for _ in scores]
        
        shap_exp = [
            [i, token, score]
            for i, (token, score) in enumerate(zip(tokens, scores))
        ]
        return shap_exp

    def explain_batch(self, texts, labels, shap_batch_size=64, normalize=True):
        """
        Explain several texts in a single SHAP call.

        Returns:
            list[list[[token_idx, token, score], ...]]
        """

        texts = [remove_chaklas(text) for text in texts]

        # One SHAP invocation for the entire batch
        with torch.inference_mode():
            shap_values = self.explainer(texts, batch_size=shap_batch_size)

        results = []

        for sample_idx, (text, label) in enumerate(zip(texts, labels)):
            tokens = self.tokenizer.tokenize(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )

            # Remove special tokens ([CLS], [SEP], etc.)
            values = shap_values.values[sample_idx][1:-1]

            polarity_idx = self.model.config.label2id[label]

            scores = values[:, polarity_idx].astype(float)

            # Defensive alignment, because SHAP/tokenizer representation
            # can occasionally differ.
            n = min(len(tokens), len(scores))
            tokens = tokens[:n]
            scores = scores[:n]

            if normalize and len(scores) > 0:
                min_score = scores.min()
                max_score = scores.max()

                if max_score != min_score:
                    scores = (
                        2 * (scores - min_score)
                        / (max_score - min_score)
                        - 1
                    )
                else:
                    scores = np.zeros_like(scores)

            results.append([
                [i, token, float(score)]
                for i, (token, score) in enumerate(zip(tokens, scores))
            ])

        return results

class IgExplainer:
    def __init__(self, model_name, device, n_steps=50):
        self.model_name = model_name
        self.device = device
        self.n_steps = n_steps
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)

        def forward_func(input_embeds, attention_mask=None):
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            # return torch.softmax(outputs.logits, dim=-1)
            return outputs.logits
        
        self.explainer = CaptumIG(forward_func)

    def explain(self, text, label, normalize=True):
        text = remove_chaklas(text)

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=self.max_length)

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)

        polarity_idx = self.model.config.label2id[label]
        attributions, _ = self.explainer.attribute(
            inputs=embeddings,
            n_steps=self.n_steps,
            baselines=torch.zeros_like(embeddings).to(self.device),
            target=polarity_idx,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
        )
        attributions = attributions.sum(dim=-1).squeeze(0)
        scores = attributions.cpu().detach().numpy()

        if normalize:
            min_score = scores.min()
            max_score = scores.max()
            if max_score != min_score:
                scores = 2 * (scores - min_score) / (max_score - min_score) - 1
            else:
                scores = np.zeros_like(scores)

        ig_exp = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            ig_exp.append([i, token, score])
        return ig_exp

class DeepLiftExplainer:
    def __init__(self, model_name, device):
        class ModelWrapper(nn.Module):
            def __init__(self, model, embedding_layer):
                super().__init__()
                self.model = model
                self.embedding_layer = embedding_layer

            def forward(self, embeddings, attention_mask=None):
                outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                # return torch.softmax(outputs.logits, dim=-1)
                return outputs.logits
            
        self.model_name = model_name
        self.device = device        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        self.embedding_layer = self.model.get_input_embeddings()
        self.wrapper_model = ModelWrapper(self.model, self.embedding_layer).to(device)
        self.explainer = DeepLift(self.wrapper_model)

    def explain(self, text, label, normalize=True):
        text = remove_chaklas(text)
        
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=self.max_length)
        input_ids = inputs["input_ids"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get embeddings and set requires_grad
        with torch.no_grad():
            embeddings = self.embedding_layer(input_ids)

        polarity_idx = self.model.config.label2id[label]

        attributions = self.explainer.attribute(
            inputs=embeddings,
            baselines=torch.zeros_like(embeddings).to(self.device),
            target=polarity_idx, # Specify the class index
            additional_forward_args=(attention_mask,)
        )
        scores = attributions.sum(dim=-1)[0].cpu().detach().numpy()
        # if normalize, normalize the scores between -1 and 1
        if normalize:
            min_score = scores.min()
            max_score = scores.max()
            if max_score != min_score:
                scores = 2 * (scores - min_score) / (max_score - min_score) - 1
            else:
                scores = np.zeros_like(scores)
        dl_exp = []
        for i, (token, score) in enumerate(zip(tokens, scores.tolist())):
            dl_exp.append([i, token, score])

        return dl_exp

class EnsembleExplainer:
    def __init__(self, mean=True, median=True, majority_sign_mean=True):
        self.mean = mean
        self.median = median
        self.majority_sign_mean = majority_sign_mean

    def explain(self, lime_results=None, shap_results=None, ig_results=None, dl_results=None):
        tokens = [lime_result[1] for lime_result in lime_results]
        lime_scores = [lime_result[2] for lime_result in lime_results] if lime_results else []
        shap_scores = [shap_result[2] for shap_result in shap_results] if shap_results else []
        ig_scores = [ig_result[2] for ig_result in ig_results] if ig_results else []
        dl_scores = [dl_result[2] for dl_result in dl_results] if dl_results else []

        exai_exp_simple_mean, exai_exp_median, = [], []
        
        for i, token in enumerate(tokens):
            scores = []

            if len(lime_scores) > 0:
                lime_score = lime_scores[i]
                scores.append(lime_score)
            if len(shap_scores) > 0:
                shap_score = shap_scores[i]
                scores.append(shap_score)
            if len(ig_scores) > 0:
                ig_score = ig_scores[i]
                scores.append(ig_score)
            if len(dl_scores) > 0:
                dl_score = dl_scores[i]
                scores.append(dl_score)

            if self.mean:
                nb_exp = len(scores)
                exai_exp_simple_mean.append([i, token, sum(scores) / nb_exp])

            if self.median:
                exai_exp_median.append([i, token, np.median(scores, axis=0)])

        return exai_exp_simple_mean, exai_exp_median