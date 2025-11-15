import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
from dataclasses import dataclass

# ============================================================================
# Model Architecture (Same as training)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ============================================================================
# Model Loading and Inference
# ============================================================================

# Initialize device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# Load model
model = None
enc = tiktoken.get_encoding('gpt2')

def load_model():
    global model
    if model is None:
        config = GPTConfig()
        model = GPT(config)
        try:
            # Try to load trained weights
            model.load_state_dict(torch.load('gpt2_model.pt', map_location=device))
            print("Loaded trained model weights")
        except FileNotFoundError:
            print("Warning: No trained weights found. Using randomly initialized model.")
            print("Please save your trained model as 'gpt2_model.pt'")
        model.to(device)
        model.eval()
    return model

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, num_sequences=1):
    """
    Generate text based on the input prompt

    Args:
        prompt: Input text to continue from
        max_length: Maximum length of generated text (in tokens)
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to sample from
        num_sequences: Number of different sequences to generate
    """
    model = load_model()

    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_sequences, 1)
    tokens = tokens.to(device)

    # Generate
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    generated_sequences = []

    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # Get last token logits

            # Apply temperature
            logits = logits / temperature

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)

            # Append to sequence
            tokens = torch.cat((tokens, xcol), dim=1)

            # Check if we've exceeded max context
            if tokens.size(1) >= model.config.block_size:
                break

    # Decode all sequences
    for i in range(num_sequences):
        generated_tokens = tokens[i].tolist()
        decoded_text = enc.decode(generated_tokens)
        generated_sequences.append(decoded_text)

    return "\n\n---\n\n".join(generated_sequences)

# ============================================================================
# Gradio Interface
# ============================================================================

def gradio_interface(prompt, max_length, temperature, top_k, num_sequences):
    """Wrapper function for Gradio interface"""
    if not prompt.strip():
        return "Please enter a prompt to generate text."

    try:
        result = generate_text(
            prompt=prompt,
            max_length=int(max_length),
            temperature=float(temperature),
            top_k=int(top_k),
            num_sequences=int(num_sequences)
        )
        return result
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="GPT-2 Text Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPT-2 Text Generator

        This is a GPT-2 language model (124M parameters) trained from scratch with optimized techniques.
        - **Final Training Loss**: 0.0891 (< 0.09 target achieved!)
        - **Training Steps**: 3,533
        - **Architecture**: 12 layers, 12 heads, 768 hidden dimensions

        Enter a text prompt and the model will generate a continuation.
        """
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="The future of artificial intelligence...",
                lines=5
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_length_slider = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Length (tokens)",
                    info="Maximum number of tokens to generate"
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, Lower = more focused"
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Number of top tokens to sample from"
                )

                num_sequences_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Sequences",
                    info="Generate multiple variations"
                )

            generate_button = gr.Button("Generate Text", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text",
                lines=20,
                show_copy_button=True
            )

    # Example prompts
    gr.Examples(
        examples=[
            ["The future of artificial intelligence is", 100, 0.8, 50, 1],
            ["Once upon a time in a land far away,", 150, 0.9, 40, 1],
            ["The key to success in machine learning", 120, 0.7, 50, 1],
            ["In the year 2050, technology will", 100, 0.8, 50, 1],
            ["The most important thing to remember is", 80, 0.7, 50, 2],
        ],
        inputs=[prompt_input, max_length_slider, temperature_slider, top_k_slider, num_sequences_slider],
        label="Example Prompts"
    )

    # Button click event
    generate_button.click(
        fn=gradio_interface,
        inputs=[prompt_input, max_length_slider, temperature_slider, top_k_slider, num_sequences_slider],
        outputs=output_text
    )

    gr.Markdown(
        """
        ---
        ### About This Model

        This GPT-2 model was trained using:
        - **Gradient Accumulation**: Effective batch size of 4,096 tokens
        - **Learning Rate Schedule**: Warmup + Cosine decay (max: 6e-4)
        - **Gradient Clipping**: Norm clipping at 1.0
        - **Optimization**: AdamW with weight decay

        **Training Performance**:
        - Achieved target loss < 0.09 in just 3,533 steps
        - Training device: Apple Silicon (MPS)
        - Throughput: ~4,300 tokens/sec

        For more details, check the [GitHub repository](https://github.com/yourusername/gpt2-training).
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
