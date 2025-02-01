import os
import torch
import transformers
from PIL import Image
from torchvision import transforms
from radiologyproj.conf import ROOT_DIR
import pathlib

ckpt_name = 'aehrc/cxrmate'

dataset_dir = ROOT_DIR.parent / 'data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_decoder = transformers.AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(device)
encoder_decoder.eval()
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(ckpt_name)
image_processor = transformers.AutoFeatureExtractor.from_pretrained(ckpt_name)

test_transforms = transforms.Compose(
    [
        transforms.Resize(size=image_processor.size['shortest_edge']),
        transforms.CenterCrop(size=[
            image_processor.size['shortest_edge'],
            image_processor.size['shortest_edge'],
        ]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std,
        ),
    ]
)
def send_prompt(images, prompt):
    outputs = encoder_decoder.generate(
        pixel_values=images.to(device),
        decoder_input_ids=prompt['input_ids'],
        special_token_ids=[
            tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index('[PMT-SEP]')
            ],
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
        ],  
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
        use_cache=False,  # Set use_cache to False
        max_length=256 + prompt['input_ids'].shape[1],
        num_beams=4,
    )

    if torch.all(outputs[:, 0] == 1):
        outputs = outputs[:, 1:]

    return outputs

def generate_caption(images, prompt):
    sequences = send_prompt(images, prompt)
    # Findings and impression sections (exclude previous report):
    _, findings, impressions = encoder_decoder.split_and_decode_sections(
        sequences,
        [tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id],
        tokenizer
    )
    return findings, impressions

def lead_and_transform(image_path):
    image = Image.open(image_path).convert('RGB')
    return test_transforms(image)

def tokenize_prompt(previous_findings, previous_impression, tokenizer, max_len, add_bos_token_id=False):
    previous_findings = ['[NPF]' if not i else i for i in previous_findings]
    previous_impression = ['[NPI]' if not i else i for i in previous_impression]
    return encoder_decoder.tokenize_prompt(
        previous_findings, 
        previous_impression, 
        tokenizer, 
        max_len, 
        add_bos_token_id=add_bos_token_id,
    )

def run_inference(image_paths: list[pathlib.Path], previous_findings, previous_impression):
    prompt = tokenize_prompt(previous_findings, previous_impression, tokenizer, 256, add_bos_token_id=True)
    images = [lead_and_transform(image_path) for image_path in image_paths]
    images = torch.stack(images, dim=0)
    ### NOTE - I need to figure out what images should be passed here ###
    ### TODO - Remove [images]*2 ###
    images = torch.nn.utils.rnn.pad_sequence([images]*2, batch_first=True, padding_value=0.0)
    return generate_caption(images, prompt)


if __name__ == '__main__':
    # No previous findings and impression sections available:
    previous_findings = [None, None]
    previous_impression = [None, None]
    image_path_1 = os.path.join(dataset_dir, 'x-ray-image-of-wrist-joint-front-view-of-normal-wrist-joint-2NM206X.jpg')
    image_path_2 = os.path.join(dataset_dir, 'photodune-2618118-hands-xray-l.JPG-nggid042182-ngg0dyn-0x0x100-00f0w010c010r110f110r010t010.jpg')
    run_inference([image_path_1, image_path_2], previous_findings, previous_impression)
