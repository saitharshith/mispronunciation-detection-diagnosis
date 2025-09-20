import torch 
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor ,Wav2Vec2Processor , pipeline, AutoModelForCTC
from datasets import Audio, load_from_disk
import evaluate
from utils.util_file import load_audio

model_path = r"C:\Users\Dell\Downloads\Final Year Project Details\saved_model"

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = AutoModelForCTC.from_pretrained(model_path)
model.eval()
blank_token_id = model.config.pad_token_id


def predicted_phonemes(input_audio):
    #get the phenomes Probabilites
    input_values = processor(input_audio , return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits , dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-2)
    return {"Predicted_Phonemes":list(transcription),"Logits":log_probs}

def calculate_gop_af(log_probs, phoneme_ids, target_phoneme_idx, blank_token_id):
    # Ensure log_probs is in log-probability space
    log_probs_for_loss = log_probs.log_softmax(dim=-1)
    
    # Permute to (Time, Batch, Classes) for CTCLoss
    log_probs_for_loss = log_probs_for_loss.permute(1, 0, 2)
    
    # Get the length of the log_probs sequence (time steps)
    input_lengths = torch.tensor([log_probs_for_loss.shape[0]])
    ctc_loss = torch.nn.CTCLoss(blank=blank_token_id, reduction='sum', zero_infinity=True)
    
    # Calculate the NLL of the ORIGINAL (CANONICAL) phoneme sequence 
    target_lengths = torch.tensor([len(phoneme_ids)])
    targets = torch.tensor(phoneme_ids, dtype=torch.long).unsqueeze(0)
    nll_canonical = ctc_loss(log_probs_for_loss, targets, input_lengths, target_lengths)

    # Calculate the NLL of the MODIFIED sequence (surrounding context) 
    modified_phoneme_ids = phoneme_ids[:target_phoneme_idx] + phoneme_ids[target_phoneme_idx+1:]
    # Handle the case where the context might be empty
    if not modified_phoneme_ids:
        nll_modified = torch.tensor(0.0)
    else:
        modified_targets = torch.tensor(modified_phoneme_ids, dtype=torch.long).unsqueeze(0)
        modified_target_lengths = torch.tensor([len(modified_phoneme_ids)])
        nll_modified = ctc_loss(log_probs_for_loss, modified_targets, input_lengths, modified_target_lengths)
    # --- 3. The GOP score is the difference in log-likelihoods ---
    gop_score = (nll_modified - nll_canonical).item()
    return gop_score
    




def get_phonetic_analysis (loaded_audio , ground_truth_text):
    
    ground_truth_phonemes = list(ground_truth_text)
    phoneme_ids = [processor.tokenizer.convert_tokens_to_ids(p) for p in ground_truth_phonemes]
    
    prediction_output = predicted_phonemes(loaded_audio)
    log_probs = prediction_output["Logits"]
    
    gop_score = []
    for i , phoneme in enumerate(ground_truth_phonemes):
        #pass predicted log probabiltes 
        score = calculate_gop_af(log_probs , phoneme_ids , i , blank_token_id)
        gop_score.append(score)
        
    return {
        "Predicted_transcription": prediction_output['Predicted_Phonemes'],
        "gop_scores": gop_score,
        "ground_truth_phonemes": ground_truth_phonemes
    }
        
        
    
    
    