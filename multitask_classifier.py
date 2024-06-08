'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# from bert_prefix_tuning import BertModel
from bert_lora3 import BertModel
from optimizer import AdamW
from tqdm import tqdm
import pynvml
import time

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you're using GPU 0

def get_gpu_usage():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)  # Convert bytes to megabytes


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_labels = config.num_labels
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model", "lora-model", "prefix-tuning-model"] # added lora model but does not work as an argument when training
        for param in self.bert.parameters():
            assert config.fine_tune_mode in ["last-linear-layer", "full-model", "lora-model"] # added lora model but does not work as an argument when training
        
        frozen_params = 0
        unfrozen_params = 0

        for name, param in self.bert.named_parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
            elif config.fine_tune_mode == 'lora-model':
                param.requires_grad = False  # Default to freezing all parameters
                if 'lora' in name or 'bias' in name or 'norm' in name or 'Norm' in name:  # Don't freeze bias, Lora, or LayerNorm
                    param.requires_grad = True  # Unfreeze specific parameters
            elif config.fine_tune_mode == 'prefix-tuning-model':
                param.requires_grad == False # freeze all pretrained parameters
                if 'prefix' in name or 'bias' in name or 'norm' in name or 'Norm' in name:  # Don't freeze bias, Lora, or LayerNorm
                    param.requires_grad = True # unfreeze prefix parameters                        
            
            if param.requires_grad:
                unfrozen_params += param.numel()  # Count individual elements
            else:
                frozen_params += param.numel()  # Count individual elements
        print(f"Number of unfrozen parameters: {frozen_params}")
        print(f"Number of total parameters: {unfrozen_params + frozen_params}")
        print(f"Parameter Reduction: {100*frozen_params/(unfrozen_params + frozen_params)}")

        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.sentiment_classifier = torch.nn.Linear(config.hidden_size, 5)
        self.paraphrase_classifier = torch.nn.Linear(config.hidden_size, 1)
        self.similarity_regressor = torch.nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        pooling = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        pooled_output = self.dropout(pooling)
        return pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooled_output = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(pooled_output)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.abs(pooled_output_1 - pooled_output_2)
        logit = self.paraphrase_classifier(combined_output)
        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        combined_output = torch.abs(pooled_output_1 - pooled_output_2)
        similarity_score = self.similarity_regressor(combined_output)
        return similarity_score




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    num_examples = min(100000, len(para_train_data))
    subset_indices = random.sample(range(len(para_train_data)), num_examples)
    para_train_data_subset = Subset(para_train_data, subset_indices)
    
    para_train_dataloader = DataLoader(para_train_data_subset, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    

    # Init model. Added Lora rank here
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    best_dev_corr = 0

    train_losses, val_losses = [], []
    train_metric, val_metric = [], []

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initialize lists to track GPU usage
    gpu_usage = []

    # keep track of parameters original state
    # initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Choose one fintuning mode
    #for dataset_name, train_dataloader, dev_dataloader in [("SST", sst_train_dataloader, sst_dev_dataloader), ("PARA", para_train_dataloader, para_dev_dataloader), ("STS", sts_train_dataloader, sts_dev_dataloader)]:
    #for dataset_name, train_dataloader, dev_dataloader in [("SST", sst_train_dataloader, sst_dev_dataloader)]:
    for dataset_name, train_dataloader, dev_dataloader in [("PARA", para_train_dataloader, para_dev_dataloader)]:
    #for dataset_name, train_dataloader, dev_dataloader in [("STS", sts_train_dataloader, sts_dev_dataloader)]:

        print(f"Training on " + dataset_name + " Dataset")
        # Run for the specified number of epochs.
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                # Record GPU usage
                gpu_usage.append(get_gpu_usage())

                if dataset_name == "SST":
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                else:
                    (b_ids1, b_mask1,
                    b_ids2, b_mask2,
                    b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                batch['token_ids_2'], batch['attention_mask_2'],
                                batch['labels'], batch['sent_ids'])

                    b_ids1 = b_ids1.to(device)
                    b_mask1 = b_mask1.to(device)
                    b_ids2 = b_ids2.to(device)
                    b_mask2 = b_mask2.to(device)
                    b_labels = b_labels.to(device).float()
                    optimizer.zero_grad()

                if dataset_name == "PARA":
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    #loss = F.binary_cross_entropy_with_logits(logits.squeeze(), b_labels.float())
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), b_labels.view(-1), reduction='sum') / args.batch_size
                elif dataset_name == "STS":
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    # Ensure logits and labels are the same shape
                    #loss = F.mse_loss(logits, b_labels.float())
                    loss = F.mse_loss(logits.squeeze(-1), b_labels.view(-1), reduction='sum') / args.batch_size


                #print(f"Logits: {logits}")
                #print(f"Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            if dataset_name == "SST":
                train_acc, dev_loss, train_f1, *_ = model_eval_sst(train_dataloader, model, device)
                dev_acc, dev_loss, dev_f1, *_ = model_eval_sst(dev_dataloader, model, device)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
                print(f"Train f1 :: {train_f1 :.3f}, dev f1 :: {dev_f1 :.3f}")
            elif dataset_name == "PARA":
                train_acc, dev_loss, train_f1, *_ = model_eval_para(train_dataloader, model, device)
                dev_acc, dev_loss, dev_f1, *_ = model_eval_para(dev_dataloader, model, device)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
                print(f"Train f1 :: {train_f1 :.3f}, dev f1 :: {dev_f1 :.3f}")
            else:
                train_corr, dev_loss, *_ = model_eval_sts(train_dataloader, model, device)
                dev_corr, dev_loss, *_ = model_eval_sts(dev_dataloader, model, device)
                if dev_corr > best_dev_corr:
                    best_dev_corr = dev_corr
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev corr :: {dev_corr :.3f}")

            train_losses.append(train_loss)
            val_losses.append(dev_loss)
            if dataset_name == "STS":
                train_metric.append(train_corr)
                val_metric.append(dev_corr)
            else:
                train_metric.append(train_acc)
                val_metric.append(dev_acc)
            save_model(model, optimizer, args, config, args.filepath)

        # Plotting Metrics
        plot_metrics(train_losses, val_losses, train_metric, val_metric, dataset_name)
        train_losses, val_losses, train_metric, val_metric = [], [], [], []


    # Calculate average and maximum GPU usage
    avg_gpu_usage = sum(gpu_usage) / len(gpu_usage)
    max_gpu_usage = max(gpu_usage)

    print(f"Average GPU usage: {avg_gpu_usage:.2f} MB")
    print(f"Maximum GPU usage: {max_gpu_usage:.2f} MB")

    """
    # Check parameter updated only Lora, bias and Layernorm
    for name, param in model.named_parameters():
        if "paraphrase" in name or "similarity" in name:
            assert torch.equal(initial_params[name], param), f"Parameter {name} changed but it should not have."
        elif "lora" in name or "bias" in name or "Norm" in name or "norm" in name or "sentiment" in name:
            assert not torch.equal(initial_params[name], param), f"Parameter {name} did not change but it should have."
        else:
            assert torch.equal(initial_params[name], param), f"Parameter {name} changed but it should not have."
    """
            

        # saving trained params
        # args.filepath = '/Users/susanahmed/Documents/GitHub/CS224N-Spring2024-DFP-Student-Handout/saved_params.pt'
        # save_model(model, optimizer, args, config, args.saved_params.pt)


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model', 'lora-model', 'prefix-tuning-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


def plot_metrics(train_losses, val_losses, train_metric, val_metric, dataset_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title(dataset_name + 'Training and Validation Loss ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric, 'bo-', label='Training Accuracy/Corr')
    plt.plot(epochs, val_metric, 'ro-', label='Validation Accuracy/Corr')
    plt.title(dataset_name + ' Training and Validation Accuracy/Corr')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Corr')
    plt.legend()

    plt.savefig(f'training_validation_metrics_{dataset_name}.png')

    plt.show()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    # HERE PUT THE NAME OF THE FILE TO SAVE PARAMETERS, Change everytime "parafinetuned"
    args.filepath = f'{args.fine_tune_mode}-PARAfinetuned-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    train_multitask(args)
    test_multitask(args)
