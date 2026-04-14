import logging
from typing import Any

logger = logging.getLogger(__name__)

_emb_model = None
_llm_model = None
_llm_tokenizer = None


def get_embedding_model():
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model...")
        _emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded (384-dim)")
    return _emb_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def get_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"Loading LLM: {model_name}...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    _llm_model.eval()
    logger.info(f"LLM loaded on {next(_llm_model.parameters()).device}")
    return _llm_model, _llm_tokenizer


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    import torch

    model, tokenizer = get_llm()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def generate_structured(
    prompt: str, schema: dict, max_new_tokens: int = 512
) -> dict[str, Any]:
    schema_str = _format_schema(schema)
    full_prompt = (
        f"{prompt}\n\n"
        f"Respond ONLY with valid JSON matching this schema:\n"
        f"{schema_str}\n\n"
        f"JSON response:"
    )
    raw = generate(full_prompt, max_new_tokens=max_new_tokens, temperature=0.1)
    return _parse_json_response(raw)


def _format_schema(schema: dict) -> str:
    import json

    example = {}
    properties = schema.get("properties", schema)
    for key, val in properties.items():
        field_type = val.get("type", "string") if isinstance(val, dict) else "string"
        if field_type == "integer":
            example[key] = 0
        elif field_type == "number":
            example[key] = 0.0
        elif field_type == "boolean":
            example[key] = False
        elif field_type == "array":
            example[key] = []
        else:
            example[key] = ""
    return json.dumps(example, indent=2)


def _parse_json_response(raw: str) -> dict[str, Any]:
    import json

    for candidate in [raw, raw.strip(), raw.strip().strip("`")]:
        if candidate.startswith("{"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    return {"raw_response": raw, "parse_error": True}
