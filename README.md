# README

This is the code repository for ResMoE of Switch Transformer and Mixtral. The checkpoint and the corresponding class will be made public once the paper got accepted.

## Switch Transformer

Use the scripts switch/finetune.py for finetuning. The model is default to 'switch-base-8'. Please specify the dataset's name or will be default to 'sst2'. For switch, please choose from the GLUE datasets (sst2, mrpc, cola, mnli).

To run the Optimal Transport switch-OT.py, you will need to load the model's weights first. Then use switch/run-ot.sh to calculate the barycenters.

After this, please evaluate the model using switch/val.py. Please make sure that the dataset you used to evaluate the model aligns with the one you used for finetuning.

## Mixtral

The process will be similar to switch, except that you do not need to finetune the model. Use python mixtral.py --k "$k" --s "$s", with 'k' being the top-k layer you want to perform the method, and 's' being the sparsity. This will automatically load WikiText, LAMBADA, WinoGrande and PIQA for zero-shot evaluate.

Note that Mixtral is a large model, so make sure you have enough space to load it. We did not perform any quantization method on it, except loading the model using bf16.

Please do load the weights first since Mixtral is extremely large.

The default script will set the initial point as the first expert per layer.

Note that in order to try the model out directly, you will need the model weights to use with `resmoe_mixtral`. However, the weights are currently too large to be uploaded as supplementary material. As a consequence, the weights will be made public through huggingface upon publication.