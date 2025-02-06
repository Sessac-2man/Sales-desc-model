from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def upload_to_model(upload_path, model_path):
    model_name = model_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
                                                             bos_token='<s>',
                                                             eos_token='</s>',
                                                             pad_token='<pad>',
                                                             unk_token='<unk>',
                                                             mask_token='<mask>'
                                                             )
    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.push_to_hub(upload_path)
    tokenizer.push_to_hub(upload_path)


if __name__ == "__main__":
    upload_to_model("Sessac-Blue/product-desc-KoGPT2-v1", "../KoGPT2-v1/output_dir")