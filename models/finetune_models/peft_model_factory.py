import os
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
try:
    from transformers import EsmTokenizer, EsmModel
except ImportError:
    # Fallback for older transformers versions
    EsmTokenizer = AutoTokenizer
    EsmModel = AutoModel

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    # BitsAndBytesConfig not available, define a dummy class
    class BitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from models.frozen_plm_model import FrozenPlmModel, FrozenFeatureExtractorModel
from models.finetune_models.peft_model import PEFTModel
from models.MTL import MTLModel
import types
import os
import glob


def load_plm_encoder(model_path, args):
    """
    Load a PLM-only model (with LoRA weights) as frozen encoder.
    This is created by remove_convbert.py script which extracts only the PLM from a trained model.
    
    Args:
        model_path: Path to the model checkpoint (.pt file), the actual adapter is in _lora folder
        args: Arguments object with model configuration
    
    Returns:
        Tuple of (plm_model, tokenizer) - the PLM model with LoRA weights (frozen)
    """
    from peft import PeftModel
    
    # Determine the actual adapter path by checking which folder exists
    # PLM encoder was created from plm-lora/plm-qlora trained models
    # We check for _lora or _qlora folder, regardless of current args.training_method
    lora_path = model_path.replace('.pt', '_lora')
    qlora_path = model_path.replace('.pt', '_qlora')
    
    if os.path.exists(lora_path):
        adapter_path = lora_path
    elif os.path.exists(qlora_path):
        adapter_path = qlora_path
    else:
        raise FileNotFoundError(
            f"PEFT adapter not found. Checked: {lora_path} and {qlora_path}. "
            f"Please ensure the PLM encoder was extracted using remove_convbert.py"
        )
    
    # Create base PLM and tokenizer
    tokenizer, base_plm_model = create_plm_and_tokenizer(args)
    
    # Load PEFT adapter
    plm_model = PeftModel.from_pretrained(base_plm_model, adapter_path)
    
    # Freeze PLM parameters
    freeze_plm_parameters(plm_model)
    
    return plm_model, tokenizer


def load_feature_extractor(model_path, args):
    """
    Load a pre-trained model without decoder as feature extractor.
    The model includes PLM (with LoRA weights) + ConvBERT (without decoder).
    
    Args:
        model_path: Path to the model checkpoint (without decoder)
        args: Arguments object with model configuration
    
    Returns:
        Tuple of (model, plm_model) - the complete feature extractor including PLM and ConvBERT
    """
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    
    from scripts.remove_last_mlp import load_model_without_decoder
    
    # Load checkpoint to infer model structure
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Infer pooling_method from checkpoint keys
    # The checkpoint still contains decoder weights (only forward is modified)
    # ConvBERTClassificationHead structure: classifier.convbert.* followed by classifier.decoder.* (MLP)
    checkpoint_keys = list(checkpoint.keys())
    
    # Check for ConvBERT: if has "classifier.convbert.", it must be ConvBERTClassificationHead
    # because convbert is always followed by decoder (MLP) in ConvBERTClassificationHead
    has_convbert = any(key.startswith("classifier.convbert.") for key in checkpoint_keys)
    
    # Check for BiLSTM: has "classifier.lstm." prefix
    has_lstm = any(key.startswith("classifier.lstm.") for key in checkpoint_keys)
    
    # Check for MLP: has "classifier.mlp." prefix (standalone MLP, not part of ConvBERT)
    has_mlp = any(key.startswith("classifier.mlp.") for key in checkpoint_keys)
    
    # Determine pooling method based on what's found
    if has_convbert:
        # ConvBERTClassificationHead: convbert.* followed by decoder.* (MLP)
        inferred_pooling_method = "convbert"
    elif has_lstm:
        inferred_pooling_method = "bilstm"
    elif has_mlp:
        inferred_pooling_method = "mlp"
    else:
        # Default to convbert (most common for feature extractors)
        inferred_pooling_method = "convbert"
    
    # Create a temporary args object matching the ORIGINAL training setup
    # The feature extractor was trained with ConvBERT, so we need to use convbert
    # even though the new model will use MLP
    import types
    temp_args = types.SimpleNamespace()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(temp_args, attr, getattr(args, attr))
    
    # Set training method to plm-lora (since the feature extractor was trained with LoRA)
    temp_args.training_method = "plm-lora"
    
    # IMPORTANT: Use inferred pooling_method to match the checkpoint structure
    # The checkpoint contains ConvBERT classifier (with decoder weights still present)
    # We need to create ConvBERT structure to load it correctly
    temp_args.pooling_method = inferred_pooling_method
    
    # Ensure LoRA parameters are set (needed for model structure creation)
    if not hasattr(temp_args, 'lora_r') or temp_args.lora_r is None:
        temp_args.lora_r = 8
    if not hasattr(temp_args, 'lora_alpha') or temp_args.lora_alpha is None:
        temp_args.lora_alpha = 24
    if not hasattr(temp_args, 'lora_dropout') or temp_args.lora_dropout is None:
        temp_args.lora_dropout = 0.1
    if not hasattr(temp_args, 'lora_target_modules') or temp_args.lora_target_modules is None:
        temp_args.lora_target_modules = ["query", "key", "value", "dense"]
    
    # Ensure pooling_dropout is set
    if not hasattr(temp_args, 'pooling_dropout') or temp_args.pooling_dropout is None:
        temp_args.pooling_dropout = 0.1
    
    # Load model using the existing function - this returns the complete model (PLM + ConvBERT without decoder)
    model, plm_model, _ = load_model_without_decoder(model_path, temp_args)
    
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'convbert'):
        return model, plm_model
    else:
        raise ValueError(f"Model at {model_path} does not have ConvBERT classifier")


def create_models(args):
    """Create and initialize models and tokenizer."""
    # Check if MTL training method
    if args.training_method == 'mtl':
        # Get group-to-tasks mapping
        group_to_tasks = get_mtl_group_to_tasks(args)
        args.group_to_tasks = group_to_tasks
        
        # Get flat task names for backward compatibility
        task_names = []
        for tasks in group_to_tasks.values():
            task_names.extend(tasks)
        args.task_names = sorted(list(set(task_names)))
        
        # Create PLM and tokenizer
        tokenizer, plm_model = create_plm_and_tokenizer(args)
        # Update hidden size based on PLM
        args.hidden_size = get_hidden_size(plm_model, args.plm_model)
        
        # Create MTL model with group-to-tasks mapping
        model = MTLModel(args, group_to_tasks)
        
        # Freeze PLM for mtl
        freeze_plm_parameters(plm_model)
        
        # Move models to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        plm_model = plm_model.to(device)
        
        return model, plm_model, tokenizer
    
    # Check if MTL with LoRA training method
    if args.training_method == 'mtl-lora':
        # Get group-to-tasks mapping
        group_to_tasks = get_mtl_group_to_tasks(args)
        args.group_to_tasks = group_to_tasks
        
        # Get flat task names for backward compatibility
        task_names = []
        for tasks in group_to_tasks.values():
            task_names.extend(tasks)
        args.task_names = sorted(list(set(task_names)))
        
        # Create PLM and tokenizer
        tokenizer, plm_model = create_plm_and_tokenizer(args)
        # Update hidden size based on PLM
        args.hidden_size = get_hidden_size(plm_model, args.plm_model)
        
        # Create MTL model with group-to-tasks mapping
        model = MTLModel(args, group_to_tasks)
        
        # Apply LoRA to PLM (same as plm-lora)
        plm_model = setup_lora_plm(plm_model, args)
        
        # Move models to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        plm_model = plm_model.to(device)
        
        return model, plm_model, tokenizer
    
    # Check if we need to load a PLM encoder (PLM-only, without classifier)
    if hasattr(args, 'plm_encoder_path') and args.plm_encoder_path:
        # Load PLM-only model (with LoRA weights) as frozen encoder
        plm_model, tokenizer = load_plm_encoder(args.plm_encoder_path, args)
        # Update hidden size based on PLM
        args.hidden_size = get_hidden_size(plm_model, args.plm_model)
        # Create model with frozen PLM encoder and new classifier
        model = FrozenPlmModel(args)
    # Check if we need to load a feature extractor (legacy, PLM + ConvBERT without decoder)
    elif hasattr(args, 'feature_extractor_path') and args.feature_extractor_path:
        # Load feature extractor (includes PLM + ConvBERT without decoder)
        feature_extractor_model, feature_extractor_plm_model = load_feature_extractor(args.feature_extractor_path, args)
        # Update hidden size based on PLM
        args.hidden_size = get_hidden_size(feature_extractor_plm_model, args.plm_model)
        # Create model with feature extractor
        model = FrozenFeatureExtractorModel(
            args, 
            feature_extractor_model=feature_extractor_model,
            feature_extractor_plm_model=feature_extractor_plm_model
        )
        # For feature extractor mode, we don't need to create a new PLM
        # But we still need tokenizer, so we create a dummy plm_model for compatibility
        # The actual PLM is already in feature_extractor_plm_model
        plm_model = feature_extractor_plm_model
        # Create tokenizer for feature extractor mode
        tokenizer, _ = create_plm_and_tokenizer(args)
    else:
        # Create standard adapter model
        tokenizer, plm_model = create_plm_and_tokenizer(args)
        # Update hidden size based on PLM
        args.hidden_size = get_hidden_size(plm_model, args.plm_model)
        model = FrozenPlmModel(args)
        # Handle PLM parameters based on training method
        freeze_plm_parameters(plm_model)
        
        if args.training_method == 'plm-lora':
            plm_model = setup_lora_plm(plm_model, args)
        elif args.training_method == 'plm-qlora':
            plm_model = create_qlora_model(plm_model, args)
        elif args.training_method == 'plm-adalora':
            plm_model = create_adalora_model(plm_model, args)
        elif args.training_method == "plm-dora":
            plm_model = create_dora_model(plm_model, args)
        elif args.training_method == "plm-ia3":
            plm_model = create_ia3_model(plm_model, args)

    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)

    return model, plm_model, tokenizer


def create_lora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = PEFTModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model = setup_lora_plm(plm_model, args)
    return model, plm_model, tokenizer


def create_qlora_model(args):
    qlora_config = setup_quantization_config()
    tokenizer, plm_model = create_plm_and_tokenizer(args, qlora_config=qlora_config)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = PEFTModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model = prepare_model_for_kbit_training(plm_model)
    plm_model = setup_lora_plm(plm_model, args)
    return model, plm_model, tokenizer


def create_dora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = PEFTModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model = setup_dora_plm(plm_model, args)
    return model, plm_model, tokenizer


def create_adalora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = PEFTModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model = setup_adalora_plm(plm_model, args)
    print(" Using plm adalora ")
    return model, plm_model, tokenizer


def create_ia3_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = PEFTModel(args=args)
    plm_model.gradient_checkpointing_enable()
    plm_model = prepare_model_for_kbit_training(plm_model)
    plm_model = setup_ia3_plm(plm_model, args)
    print(" Using plm IA3 ")
    return model, plm_model, tokenizer


def peft_factory(args):
    if args.training_method == "plm-lora":
        model, plm_model, tokenizer = create_lora_model(args)
    elif args.training_method == "plm-qlora":
        model, plm_model, tokenizer = create_qlora_model(args)
    elif args.training_method == "plm-dora":
        model, plm_model, tokenizer = create_dora_model(args)
    elif args.training_method == "plm-adalora":
        model, plm_model, tokenizer = create_adalora_model(args)
    elif args.training_method == "plm-ia3":
        model, plm_model, tokenizer = create_ia3_model(args)
    else:
        raise ValueError(f"Unsupported PEFT training method: {args.training_method}")
    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)
    return model, plm_model, tokenizer


def freeze_plm_parameters(plm_model):
    """Freeze all parameters in the pre-trained language model."""
    for param in plm_model.parameters():
        param.requires_grad = False
    plm_model.eval()  # Set to evaluation mode


def setup_quantization_config():
    """Setup quantization configuration."""
    from transformers import BitsAndBytesConfig
    # https://huggingface.co/docs/peft/v0.14.0/en/developer_guides/quantization#quantize-a-model
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return qlora_config


def setup_lora_plm(plm_model, args):
    """Setup LoRA for pre-trained language model."""
    # Import LoRA configurations
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    # Apply LoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model


def setup_dora_plm(plm_model, args):
    """Setup DoRA for pre-trained language model."""
    # Import DoRA configurations
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate Dora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure DoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        use_dora=True
    )
    # Apply DoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model


def setup_adalora_plm(plm_model, args):
    """Setup AdaLoRA for pre-trained language model."""
    # Import AdaLoRA configurations
    from peft import get_peft_config, get_peft_model, AdaLoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")

    # Calculate total steps for AdaLoRA
    # Assuming we have train_loader available, but we'll use a reasonable estimate
    # For sequence labeling tasks, typically 150 epochs with reasonable batch sizes
    estimated_steps_per_epoch = 100  # This is a reasonable estimate
    total_steps = args.num_epochs * estimated_steps_per_epoch

    # Configure AdaLoRA
    peft_config = AdaLoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        peft_type="ADALORA",
        init_r=12,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        total_step=total_steps
    )
    # Apply AdaLoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model


def setup_ia3_plm(plm_model, args):
    """Setup IA3 for pre-trained language model."""
    # Import LoRA configurations
    from peft import IA3Model, IA3Config, get_peft_model, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    print(available_modules)
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure LoRA
    peft_config = IA3Config(
        task_type=TaskType.FEATURE_EXTRACTION,
        peft_type="IA3",
        target_modules=args.lora_target_modules,
        feedforward_modules=args.feedforward_modules
    )
    # Apply LoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model


def create_plm_and_tokenizer(args, qlora_config=None):
    """Create pre-trained language model and tokenizer based on model type."""
    if "esm" in args.plm_model:
        try:
            print(f"Loading tokenizer from {args.plm_model}...")
            tokenizer = EsmTokenizer.from_pretrained(args.plm_model, trust_remote_code=True)
            print(f"Loading model from {args.plm_model}...")
            
            # Temporarily rename safetensors file to force use pytorch_model.bin
            import os
            safetensors_path = os.path.join(args.plm_model, "model.safetensors")
            safetensors_backup = os.path.join(args.plm_model, "model.safetensors.backup")
            if os.path.exists(safetensors_path):
                print(f"Temporarily renaming {safetensors_path} to avoid safetensors loading issues")
                os.rename(safetensors_path, safetensors_backup)
            
            try:
                if qlora_config:
                    plm_model = EsmModel.from_pretrained(
                        args.plm_model, 
                        quantization_config=qlora_config, 
                        trust_remote_code=True,
                        add_pooling_layer=False  # Disable pooler to avoid quantization issues
                    )
                else:
                    plm_model = EsmModel.from_pretrained(
                        args.plm_model, 
                        trust_remote_code=True,
                        add_pooling_layer=False  # Disable pooler to avoid quantization issues
                    )
                print("Model and tokenizer loaded successfully!")
            finally:
                # Restore safetensors file if it was renamed
                if os.path.exists(safetensors_backup):
                    print(f"Restoring {safetensors_backup}")
                    os.rename(safetensors_backup, safetensors_path)
                    
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            print("Trying to download the model...")
            try:
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True, use_fast=False)
                if qlora_config:
                    plm_model = AutoModel.from_pretrained(
                        args.plm_model, 
                        quantization_config=qlora_config, 
                        trust_remote_code=True,
                        add_pooling_layer=False  # Disable pooler to avoid quantization issues
                    )
                else:
                    plm_model = AutoModel.from_pretrained(
                        args.plm_model, 
                        trust_remote_code=True,
                        add_pooling_layer=False  # Disable pooler to avoid quantization issues
                    )
                print("Model and tokenizer downloaded and loaded successfully!")
            except Exception as e2:
                print(f"Failed to download model: {e2}")
                raise ValueError(f"Cannot load model {args.plm_model}. Please check the model name or internet connection.")
    elif "ankh" in args.plm_model.lower():
        try:
            print(f"Loading Ankh base tokenizer from {args.plm_model}...")
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True, use_fast=False)
            print(f"Loading Ankh base model from {args.plm_model}...")
            if qlora_config:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    quantization_config=qlora_config, 
                    trust_remote_code=True
                )
            else:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    trust_remote_code=True
                )
            print("Ankh base model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading Ankh base model/tokenizer: {e}")
            raise ValueError(f"Cannot load Ankh base model {args.plm_model}. Please check the model name or internet connection.")
    elif "prott5" in args.plm_model.lower() or "prot_t5" in args.plm_model.lower():
        try:
            print(f"Loading ProtT5 tokenizer from {args.plm_model}...")
            from transformers import AutoTokenizer, T5EncoderModel
            tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True, use_fast=False)
            print(f"Loading ProtT5 model from {args.plm_model}...")
            if qlora_config:
                plm_model = T5EncoderModel.from_pretrained(
                    args.plm_model, 
                    quantization_config=qlora_config, 
                    trust_remote_code=True
                )
            else:
                plm_model = T5EncoderModel.from_pretrained(
                    args.plm_model, 
                    trust_remote_code=True
                )
            print("ProtT5 model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading ProtT5 model/tokenizer: {e}")
            raise ValueError(f"Cannot load ProtT5 model {args.plm_model}. Please check the model name or internet connection.")
    elif "protbert" in args.plm_model.lower() or "prot_bert" in args.plm_model.lower():
        try:
            print(f"Loading ProtBert tokenizer from {args.plm_model}...")
            from transformers import AutoTokenizer, AutoModel
            
            # Use custom tokenizer for ProtBERT
            class ProteinTokenizer:
                def __init__(self):
                    self.vocab = {
                        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                        "L": 5, "A": 6, "G": 7, "V": 8, "E": 9, "S": 10,
                        "I": 11, "K": 12, "R": 13, "D": 14, "T": 15, "P": 16,
                        "N": 17, "Q": 18, "F": 19, "Y": 20, "M": 21, "H": 22,
                        "C": 23, "W": 24, "X": 25, "U": 26, "B": 27, "Z": 28, "O": 29
                    }
                    self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
                    self.cls_token = "[CLS]"
                    self.sep_token = "[SEP]"
                    self.pad_token = "[PAD]"
                    self.unk_token = "[UNK]"
                    self.mask_token = "[MASK]"
                    self.cls_token_id = 2
                    self.sep_token_id = 3
                    self.pad_token_id = 0
                    self.unk_token_id = 1
                    self.mask_token_id = 4
                    self.add_special_tokens = True
                    
                def tokenize(self, text):
                    tokens = []
                    for char in text:
                        if char in self.vocab:
                            tokens.append(char)
                        else:
                            tokens.append(self.unk_token)
                    return tokens
                    
                def convert_tokens_to_ids(self, tokens):
                    return [self.vocab.get(token, self.unk_token_id) for token in tokens]
                    
                def convert_ids_to_tokens(self, ids):
                    return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
                    
                def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, **kwargs):
                    if isinstance(text, str):
                        text = [text]
                    
                    all_input_ids = []
                    all_attention_masks = []
                    
                    for seq in text:
                        tokens = self.tokenize(seq)
                        if self.add_special_tokens:
                            tokens = [self.cls_token] + tokens + [self.sep_token]
                        
                        input_ids = self.convert_tokens_to_ids(tokens)
                        if truncation and max_length and len(input_ids) > max_length:
                            input_ids = input_ids[:max_length-1] + [self.sep_token_id]
                        
                        attention_mask = [1] * len(input_ids)
                        all_input_ids.append(input_ids)
                        all_attention_masks.append(attention_mask)
                    
                    if padding:
                        max_len = max(len(ids) for ids in all_input_ids)
                        for i in range(len(all_input_ids)):
                            while len(all_input_ids[i]) < max_len:
                                all_input_ids[i].append(self.pad_token_id)
                                all_attention_masks[i].append(0)
                    
                    result = {"input_ids": all_input_ids, "attention_mask": all_attention_masks}
                    
                    if return_tensors == "pt":
                        import torch
                        result["input_ids"] = torch.tensor(result["input_ids"])
                        result["attention_mask"] = torch.tensor(result["attention_mask"])
                    
                    return result
            
            tokenizer = ProteinTokenizer()
            print(f"Loading ProtBert model from {args.plm_model}...")
            if qlora_config:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    quantization_config=qlora_config, 
                    trust_remote_code=True
                )
            else:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    trust_remote_code=True
                )
            print("ProtBert model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading ProtBert model/tokenizer: {e}")
            raise ValueError(f"Cannot load ProtBert model {args.plm_model}. Please check the model name or internet connection.")
    else:
        # Generic AutoModel/AutoTokenizer support for other models
        try:
            print(f"Loading tokenizer from {args.plm_model}...")
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True, use_fast=False)
            print(f"Loading model from {args.plm_model}...")
            if qlora_config:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    quantization_config=qlora_config, 
                    trust_remote_code=True
                )
            else:
                plm_model = AutoModel.from_pretrained(
                    args.plm_model, 
                    trust_remote_code=True
                )
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            raise ValueError(f"Cannot load model {args.plm_model}. Please check the model name or internet connection.")

    return tokenizer, plm_model


def get_hidden_size(plm_model, model_type):
    """Get hidden size based on model type."""
    if "esm" in model_type:
        return plm_model.config.hidden_size
    elif "ankh" in model_type.lower():
        # Ankh base model hidden size
        return plm_model.config.hidden_size
    elif "prott5" in model_type.lower() or "prot_t5" in model_type.lower():
        # ProtT5 model hidden size
        return plm_model.config.hidden_size
    elif "protbert" in model_type.lower() or "prot_bert" in model_type.lower():
        # ProtBert model hidden size
        return plm_model.config.hidden_size
    else:
        # Generic approach for other models that have hidden_size in config
        if hasattr(plm_model.config, 'hidden_size'):
            return plm_model.config.hidden_size
        elif hasattr(plm_model.config, 'd_model'):
            return plm_model.config.d_model
        else:
            raise ValueError(f"Cannot determine hidden size for model type: {model_type}. "
                           f"Please check the model configuration.")


def get_mtl_group_to_tasks(args):
    """
    Get group-to-tasks mapping for MTL from data_group directory structure.
    Each group (category) shares an encoder, tasks within a group share the encoder.
    
    Args:
        args: Arguments object with data_group_dir and optionally mtl_group
    
    Returns:
        Dictionary mapping group names to task names
        e.g., {'nucleotide': ['ADP', 'ATP', 'AMP', 'ANP']} if mtl_group='nucleotide'
        or {'nucleotide': ['ADP', 'ATP', 'AMP', 'ANP'], 
            'inorganic_ion': ['CO', 'PO4', 'SO4'],
            'cofactor': ['FAD', 'NAD', 'SAH', 'SF4']} if mtl_group is None
    """
    data_group_dir = getattr(args, 'data_group_dir', 'dataset/data_group')
    mtl_group = getattr(args, 'mtl_group', None)
    
    if not os.path.exists(data_group_dir):
        raise ValueError(f"Data group directory not found: {data_group_dir}")
    
    # Discover tasks grouped by category
    # Structure: data_group/{category}/{molecule}/{train,val,test}.csv
    group_to_tasks = {}
    for category_dir in os.listdir(data_group_dir):
        category_path = os.path.join(data_group_dir, category_dir)
        if os.path.isdir(category_path):
            # If mtl_group is specified, only process that group
            if mtl_group is not None and category_dir != mtl_group:
                continue
                
            tasks = []
            for molecule_dir in os.listdir(category_path):
                molecule_path = os.path.join(category_path, molecule_dir)
                if os.path.isdir(molecule_path):
                    # Check if it has train.csv
                    train_file = os.path.join(molecule_path, 'train.csv')
                    if os.path.exists(train_file):
                        tasks.append(molecule_dir)
            
            if tasks:
                # Sort tasks within each group
                group_to_tasks[category_dir] = sorted(tasks)
    
    if not group_to_tasks:
        if mtl_group is not None:
            raise ValueError(f"No tasks found for group '{mtl_group}' in {data_group_dir}.")
        else:
            raise ValueError(f"No tasks found in {data_group_dir}. "
                            f"Expected structure: {data_group_dir}/{{category}}/{{molecule}}/train.csv")
    
    return group_to_tasks


def get_mtl_task_names(args):
    """
    Get flat list of task names for MTL (for backward compatibility).
    
    Args:
        args: Arguments object with data_group_dir
    
    Returns:
        List of task names (molecule types)
    """
    group_to_tasks = get_mtl_group_to_tasks(args)
    # Flatten all task names
    task_names = []
    for tasks in group_to_tasks.values():
        task_names.extend(tasks)
    return sorted(list(set(task_names))) 