import os
import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from models.frozen_plm_model import FrozenPlmModel
from models.finetune_models.peft_model import PEFTModel


def create_models(args):
    """Create and initialize models and tokenizer."""
    # Create tokenizer and PLM
    tokenizer, plm_model = create_plm_and_tokenizer(args)

    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)

    # Create adapter model
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

    # Configure AdaLoRA
    peft_config = AdaLoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        peft_type="ADALORA",
        init_r=12,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
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
            if qlora_config:
                plm_model = EsmModel.from_pretrained(args.plm_model, quantization_config=qlora_config, trust_remote_code=True)
            else:
                plm_model = EsmModel.from_pretrained(args.plm_model, trust_remote_code=True)
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            print("Trying to download the model...")
            try:
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True)
                if qlora_config:
                    plm_model = AutoModel.from_pretrained(args.plm_model, quantization_config=qlora_config, trust_remote_code=True)
                else:
                    plm_model = AutoModel.from_pretrained(args.plm_model, trust_remote_code=True)
                print("Model and tokenizer downloaded and loaded successfully!")
            except Exception as e2:
                print(f"Failed to download model: {e2}")
                raise ValueError(f"Cannot load model {args.plm_model}. Please check the model name or internet connection.")
    else:
        raise ValueError(f"Unsupported model type: {args.plm_model}")

    return tokenizer, plm_model


def get_hidden_size(plm_model, model_type):
    """Get hidden size based on model type."""
    if "esm" in model_type:
        return plm_model.config.hidden_size
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 