import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import pickle
import argparse
from datetime import datetime


def create_BERT_embeddings(sentences, question_ids, model_name, save_name, save):
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # encode
    encoded = tokenizer(sentences,
                        padding=True, truncation=True, return_tensors="pt")

    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(model_name,
                                      output_hidden_states=True)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask)
        hidden_states = outputs[2]

    embeddings_ls = []
    for i in range(len(sentences)):
        layer_last_two = hidden_states[-2][i]
        ave_layser_last_two = torch.mean(layer_last_two, dim=0)
        embeddings_ls.append(ave_layser_last_two.numpy())

    embedding_df = pd.DataFrame(data=embeddings_ls,
                                columns=["dim%d" % (i + 1) for i in range(len(ave_layser_last_two.numpy()))])

    # add the name of the sentence as the first column
    embedding_df.insert(loc=0, column='question_id', value=question_ids)

    # save or not
    if save:
        now = datetime.now()
        now = now.strftime("%Y%m%d%H%M%S")
        save_path = './data/embeddings/' + save_name + '_' + model_name.replace("-", "_") + "_" + now + '.pkl'
        embedding_df.to_pickle(save_path,
                               protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile",
                        type=str,
                        default=None)
    parser.add_argument("--model",
                        type=str,
                        default=None)
    parser.add_argument("--savename",
                        type=str,
                        default=None)
    parser.add_argument("--save",
                        type=str,
                        default=True)

    args = parser.parse_args()

    model = args.model
    save = args.save
    savename = args.savename

    if "synthetic" in savename or "Synthetic" in savename:
        datafile = './data/synthetic/' + args.datafile

    elif "ESS" in savename or "ess" in savename:
        datafile = './data/ESS/' + args.datafile

    else:
        print("Wrong data file name. Should contain 'synthetic'")
        exit()

    if "xlsx" in datafile:
        questions_df = pd.read_excel(datafile)
    else:
        print("Only excel data files are supported.")
        exit()

    if "synthetic" in savename or "Synthetic" in savename:
        questions = questions_df.rfa.to_list()
        question_names = questions_df.row_id.to_list()

    elif "ESS" in savename or "ess" in savename:
        questions = questions_df.question_UK.to_list()
        question_names = questions_df.name.to_list()

    create_BERT_embeddings(sentences=questions,
                           question_ids=question_names,
                           model_name=model,
                           save_name=savename,
                           save=save)

if __name__ == '__main__':
    main()