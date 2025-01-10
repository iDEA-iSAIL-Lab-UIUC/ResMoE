import evaluate
import torch
import wandb
import torch.nn.utils.prune as prune
import torch.nn as nn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

import sys
sys.path.append('../')

import torch

from datasets import load_dataset, load_metric

from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

from tqdm import tqdm

import torch.nn.utils.prune as prune
import torch.nn as nn




  
def train(epoch, model, device, loader, optimizer, scheduler,logging, wandb):
    model.train()
    for _,data in enumerate(loader, 0):
        # data = {k: v.cuda() for k, v in data.items()}        
        # with accelerator.accumulate(model):
        
          # labels = data['target_ids']        
          # labels = labels.masked_fill_(labels == 0, -100)
          # ids = data['source_ids']
          # mask = data['source_mask']
          
          # print(labels.device,ids.device,mask.device)          
          
        labels = data['target_ids'].to(device, dtype = torch.long)
        
        labels = labels.masked_fill_(labels == 0, -100)
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        # print(labels.shape,ids.shape,mask.shape)
        # decoder_input_ids = torch.zeros_like(labels).long()
        
        # with torch.autocast("cuda"):
            # outputs = model(**eval_batch, output_router_logits=True, return_dict=True)    
        # outputs = model(input_ids = ids, attention_mask = mask, labels=labels)    
        outputs = model(input_ids = ids, attention_mask = mask, labels=labels, output_router_logits=True, return_dict=True)
        train_loss = outputs[0]      
        wandb.log({"train_loss": train_loss.item()})    
        
          
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
      
    logging.info(f"Epoch: {epoch}, Train loss: {train_loss}")
  
  
def validate(epoch, tokenizer, model, device, loader,OUT_MAX_LEN):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids_y'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=OUT_MAX_LEN, 
                # num_beams=2,
                # repetition_penalty=2.5, 
                # length_penalty=1.0, 
                # early_stopping=True
                )
            # print(y)
            # print(generated_ids)            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            # target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            target = [t.item() for t in y]
            
            
            # print(preds,target)
            if _!=0 and _%100==0:
                print(f'Completed {_}')
                # break

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
  
  
  
def get_continuation_logits(model, tokenizer, context, continuation):
    # print(context,"\n",continuation)
    # Tokenize and encode the context and continuation
    context_tokens = tokenizer.encode(context, return_tensors='pt')
    continuation_tokens = tokenizer.encode(continuation, return_tensors='pt')
    
    # print(context_tokens.shape,continuation_tokens.shape)
    # print(context_tokens,continuation_tokens)

    # Print token lengths for diagnostic purposes

    # Combine the tokens (without duplicating the overlapping token)
    input_tokens = torch.cat((context_tokens, continuation_tokens[:,1:]), dim=1)[:,:-1]
    # input_tokens = torch.cat((context_tokens, continuation_tokens), dim=1)[:,:-1]
    
    # print(input_tokens,input_tokens.shape)
    # print(input_tokens,tokenizer.decode(input_tokens[0]))

    # Get model output
    with torch.no_grad():
        outputs = model(input_tokens)        
        multi_logits = F.log_softmax(
                    outputs.logits, dim=-1
                )
    
    # print(multi_logits)
    # Extract the logits for the continuation tokens
    continuation_start = context_tokens.size(1) - 1
    continuation_logits = multi_logits[0, continuation_start:, :].unsqueeze(0)
    
    # print(input_tokens,input_tokens[:,continuation_start:],continuation_logits.shape)
    
    
    # continuation_logits = torch.gather(continuation_logits, 2, 
    #                                    continuation_tokens[:,0:].unsqueeze(-1)).squeeze(-1)  # [1, seq]
    
    continuation_logits = torch.gather(continuation_logits, 2, 
                                       continuation_tokens[:,1:].unsqueeze(-1)).squeeze(-1)
    
    # print(continuation_logits,float(continuation_logits.sum()))
    # del 
    
    return float(continuation_logits.sum())
  
  
def batch_get_continuation_logits(model, tokenizer, contexts, continuations):
    loss = []
    # print(contexts,continuations)
    # Tokenize and encode the contexts and continuations in batch with padding
    input_ids,attention_mask = [],[]
    conlen, ctlen = [],[]
    inlen = []
    continuation_tokens= []
    for i in range(len(contexts)):
      context_input = tokenizer(contexts[i],return_tensors='pt')
      continuation_input = tokenizer(continuations[i],return_tensors='pt')
      # print(context_input,continuation_input)
      conlen.append(context_input['input_ids'].size(1))
      ctlen.append(continuation_input['input_ids'].size(1)) 
      
      continuation_tokens.append(continuation_input['input_ids'])   
      
      input_token = torch.cat((context_input['input_ids'], continuation_input['input_ids'][:,1:]), dim=1)[:,:-1]
      input_ids.append(input_token[0])
      inlen.append(len(input_token[0]))
      
    # print(input_ids,conlen,ctlen)
    # inlen = [conlen[i]+ctlen[i] for i in range(len(conlen))]
    pad = max(inlen)
    # print(continuation_tokens)
    # continuation_tokens = torch.stack(continuation_tokens)
    # print(inlen)
    
    for i in range(len(input_ids)):
      if inlen[i] == pad:
        attention_mask.append(torch.ones(pad,dtype=int))
        continue
      
      input_ids[i] = torch.cat((input_ids[i],torch.full((pad-inlen[i],),tokenizer.pad_token_id,dtype=int)),dim=0)
      
      # print(torch.cat((torch.ones(inlen[i],dtype=int),torch.zeros(pad-inlen[i],dtype=int)),dim=0))
      attention_mask.append(torch.cat((torch.ones(inlen[i],dtype=int),torch.zeros(pad-inlen[i],dtype=int)),dim=0))
      

  
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    # print(input_ids,attention_mask)

    # return logits_sums
    with torch.no_grad():
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)        
        multi_logits = F.log_softmax(
                    outputs.logits, dim=-1
                )
    
    # print(multi_logits.shape)
    
    for i in range(len(conlen)):
      continuation_logits = multi_logits[i][ conlen[i]-1:, :].unsqueeze(0)
      # print(continuation_input)
      
      # print(input_ids[i],input_ids[i,conlen[i]-1:],continuation_logits.shape)
      
      
      # continuation_logits = torch.gather(continuation_logits, 2, 
      #                                    continuation_tokens[:,0:].unsqueeze(-1)).squeeze(-1)  # [1, seq]
      
      continuation_logits = torch.gather(continuation_logits, 2, 
                                        continuation_tokens[i][:,1:].unsqueeze(-1)).squeeze(-1)
      
      # print(continuation_logits)
      loss.append(float(continuation_logits.sum()))
      
    
    # print(continuation_logits,float(continuation_logits.sum()))
    # del 
    del input_ids,attention_mask,context_input,continuation_input,input_token,outputs,multi_logits,continuation_logits
    torch.cuda.empty_cache()
    
    
    return loss
  
  
  
def batch_get_continuation_tokens(model, tokenizer, contexts, continuations):
  
    ans = []
    # print(contexts,continuations)
  
    # Tokenize and encode the contexts and continuations in batch with padding
    input_ids,attention_mask = [],[]
    conlen, ctlen = [],[]
    inlen = []
    continuation_tokens= []
    for i in range(len(contexts)):
      context_input = tokenizer(contexts[i],return_tensors='pt')
      continuation_input = tokenizer(continuations[i],return_tensors='pt')
      # print(context_input,continuation_input)
      conlen.append(context_input['input_ids'].size(1))
      ctlen.append(continuation_input['input_ids'].size(1)) 
      
      # print(context_input,continuation_input)
      
      continuation_tokens.append(continuation_input['input_ids'])  
      
      input_token = torch.cat((context_input['input_ids'], continuation_input['input_ids'][:,1:]), dim=1)[:,:-1]
      
      # input_token = torch.cat((context_input['input_ids'], continuation_input['input_ids'][:,1:]), dim=1)[:,:-1]
      input_ids.append(input_token[0])
      inlen.append(len(input_token[0]))
      
    # print(input_ids,conlen,ctlen)
    # inlen = [conlen[i]+ctlen[i] for i in range(len(conlen))]
    pad = max(inlen)
    # print(continuation_tokens)
    # continuation_tokens = torch.stack(continuation_tokens)
    # print(inlen)
    
    for i in range(len(input_ids)):
      if inlen[i] == pad:
        attention_mask.append(torch.ones(pad,dtype=int))
        continue
      
      input_ids[i] = torch.cat((input_ids[i],torch.full((pad-inlen[i],),tokenizer.pad_token_id,dtype=int)),dim=0)
      
      # print(torch.cat((torch.ones(inlen[i],dtype=int),torch.zeros(pad-inlen[i],dtype=int)),dim=0))
      attention_mask.append(torch.cat((torch.ones(inlen[i],dtype=int),torch.zeros(pad-inlen[i],dtype=int)),dim=0))

  
    input_ids = torch.stack(input_ids).to(device)
    attention_mask = torch.stack(attention_mask).to(device)
    # print(input_ids.shape,attention_mask)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        probabilities = F.log_softmax(outputs.logits, dim=-1) 
        
    # print(input_ids.shape)
    
    for i in range(len(probabilities)):
      
      greedy_tokens = (probabilities[i][conlen[i]-1:inlen[i], :].argmax(dim=-1)).to('cpu')
      # print(greedy_tokens,continuation_tokens[i][0][1:])
      
      max_equal = (greedy_tokens == continuation_tokens[i][0][1:]).all()
    
      # print(max_equal)
      ans.append(max_equal.item())
    
    
    return ans

  
  
  
def ppl(model, encodings):
  max_length = 2048
  stride = 2048
  seq_len = encodings.input_ids.size(1)

  nlls = []
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride)):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)
          # loss is calculated using CrossEntropyLoss which averages over valid labels
          # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
          # to the left by 1.
          neg_log_likelihood = outputs.loss

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break

  ppl = torch.exp(torch.stack(nlls).mean())
  
  return round(ppl.item(),2)


def evaluate_model(model,tokenizer,dataset,dataset_name, batch_size=32, fewshot=None):
    correct = 0
    total = 0

    if dataset_name == "winogrande":
    # Process the dataset in batches
      for i in range(0, len(dataset), batch_size):
          batch = dataset[i:i + batch_size]

          answers=batch['answer']

          # Calculate likelihood for each option in the batch
          likelihoods_option1 = batch_get_continuation_logits(model,tokenizer,batch["context1"], batch["cont"])
          likelihoods_option2 = batch_get_continuation_logits(model,tokenizer,batch["context2"], batch["cont"])
          
          # print(likelihoods_option1,likelihoods_option2)

          # Determine predictions and update correct count
          for j in range(len(likelihoods_option1)):
              prediction = '1' if likelihoods_option1[j] > likelihoods_option2[j] else '2'
              if prediction == answers[j]:
                  correct += 1
              total += 1
              
          print(i)

    elif dataset_name == "hellaswag":
      for i in range(0, len(dataset), batch_size):
          batch = dataset[i:i + batch_size]

          answers=batch['label']

          # Calculate likelihood for each option in the batch
          likelihoods_option1 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["end1"])
          likelihoods_option2 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["end2"])
          likelihoods_option3 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["end3"])
          likelihoods_option4 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["end4"])
          
          print(likelihoods_option1,likelihoods_option2,likelihoods_option3,likelihoods_option4)
          
          # Determine predictions and update correct count
          for j in range(batch_size):
              # print(j)
              opt_list = [likelihoods_option1[j],likelihoods_option2[j],likelihoods_option3[j],likelihoods_option4[j]]
              prediction = str(opt_list.index(max(opt_list)))
              print(opt_list,prediction)
              if prediction == answers[j]:
                  correct += 1
              total += 1
              
          # return

    elif dataset_name == "piqa":
      for i in range(0, len(dataset), batch_size):
          batch = dataset[i:i + batch_size]

          answers=batch['label']

          # Calculate likelihood for each option in the batch
          likelihoods_option1 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["sol1"])
          likelihoods_option2 = batch_get_continuation_logits(model,tokenizer,batch["context"], batch["sol2"])
          
          # print(likelihoods_option1,likelihoods_option2)
          
          # Determine predictions and update correct count
          for j in range(len(likelihoods_option1)):
              # print(j)
              opt_list = [likelihoods_option1[j],likelihoods_option2[j]]
              prediction = opt_list.index(max(opt_list))
              # print(opt_list,prediction)
              if prediction == answers[j]:
                  correct += 1
              total += 1

          print(i)
          # return 
      
    elif dataset_name == "lambada":
      for i in range(0, len(dataset), batch_size):
          batch = dataset[i:i + batch_size]


          ans = batch_get_continuation_tokens(model,tokenizer,batch['context'],batch['answer'])
          
          correct += sum(ans)
          total += len(ans)

          print(i)  
      
    accuracy = correct / total
    return round(accuracy,4)
