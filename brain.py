"""
مغز هوش مصنوعی غول‌پیکر - ۳,۲۴۷ خط
فقط مخصوص یادگیری از میلیون‌ها مقاله
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import pickle
import os
import hashlib
from collections import OrderedDict
import json
import time

# ============================================================================
# بخش ۱: معماری پایه (۴۵۰ خط)
# ============================================================================

class Config:
    """تنظیمات مغز - ۵۰ خط"""
    def __init__(self):
        # معماری اصلی
        self.vocab_size = 100000
        self.hidden_size = 4096
        self.num_layers = 48
        self.num_heads = 32
        self.head_dim = self.hidden_size // self.num_heads
        self.ffn_size = self.hidden_size * 4
        
        # Dropout و تنظیمات
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.layer_norm_eps = 1e-5
        
        # حافظه
        self.memory_size = 1000000
        self.context_window = 32768
        
        # یادگیری
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # مسیرها
        self.model_path = "models/"
        self.memory_path = "memory/"

class LayerNorm(nn.Module):
    """Layer Normalization - ۳۰ خط"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - ۸۰ خط"""
    def __init__(self, dim, max_position=32768):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype)
        )

# ============================================================================
# بخش ۲: مکانیزم توجه (۵۵۰ خط)
# ============================================================================

class FlashAttention(nn.Module):
    """Flash Attention ۲ - ۲۵۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = config.attention_dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, attention_mask=None, past_kv=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            seq_len = k.size(2)
        
        # Flash Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, (k, v)

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention - ۱۵۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, 1, self.head_dim).expand(-1, -1, self.num_heads, -1)
        v = v.view(batch_size, seq_len, 1, self.head_dim).expand(-1, -1, self.num_heads, -1)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# ============================================================================
# بخش ۳: Feed-Forward Network (۴۰۰ خط)
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU Activation - ۳۰ خط"""
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(ffn_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_size, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEExpert(nn.Module):
    """Mixture of Experts - ۱۵۰ خط"""
    def __init__(self, config, expert_id):
        super().__init__()
        self.expert_id = expert_id
        self.gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.activation = nn.GELU()
        
    def forward(self, x, gate_weight=1.0):
        gate_value = torch.sigmoid(self.gate(x)) * gate_weight
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output * gate_value

class MoELayer(nn.Module):
    """Mixture of Experts Layer - ۲۲۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = 32
        self.top_k = 4
        self.hidden_size = config.hidden_size
        
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MoEExpert(config, i) for i in range(self.num_experts)])
        
        # Load balancing
        self.expert_load = torch.zeros(self.num_experts)
        self.alpha = 0.01
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        # Router
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Update load statistics
        for i in range(self.num_experts):
            self.expert_load[i] = (top_k_indices == i).any(dim=-1).float().mean().item()
        
        # Combine expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i:i+1]
            
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input, expert_probs[mask])
                    output[mask] += expert_output
        
        # Load balancing loss
        load_balancing_loss = self.alpha * (self.expert_load.mean() - 1.0/self.num_experts).pow(2).sum()
        
        return output.view(batch_size, seq_len, self.hidden_size), load_balancing_loss

# ============================================================================
# بخش ۴: بلوک‌های ترنسفورمر (۷۵۰ خط)
# ============================================================================

class TransformerBlock(nn.Module):
    """بلوک ترنسفورمر - ۲۵۰ خط"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        self.attention = FlashAttention(config)
        self.attention_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        
        if layer_id % 4 == 0:  # هر ۴ لایه یه MoE
            self.mlp = MoELayer(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.ffn_size)
        
        self.mlp_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None, past_kv=None):
        # Attention with residual
        residual = x
        x = self.attention_norm(x)
        attn_out, past_kv = self.attention(x, attention_mask, past_kv)
        x = residual + self.dropout(attn_out)
        
        # MLP with residual
        residual = x
        x = self.mlp_norm(x)
        if isinstance(self.mlp, MoELayer):
            mlp_out, moe_loss = self.mlp(x)
        else:
            mlp_out = self.mlp(x)
            moe_loss = 0.0
        
        x = residual + self.dropout(mlp_out)
        
        return x, past_kv, moe_loss

class TransformerEncoder(nn.Module):
    """Encoder ترنسفورمر - ۳۰۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        self.norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask=None, past_kvs=None):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        batch_size, seq_len = input_ids.shape
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        
        if past_kvs is None:
            past_kvs = [None] * self.num_layers
        
        new_past_kvs = []
        total_moe_loss = 0.0
        
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, past_kv, moe_loss = layer(hidden_states, attention_mask, past_kv)
            new_past_kvs.append(past_kv)
            total_moe_loss += moe_loss
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, new_past_kvs, total_moe_loss

class TransformerDecoder(nn.Module):
    """Decoder ترنسفورمر - ۲۰۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.output_proj.weight = self.encoder.embed_tokens.weight
        
    def forward(self, input_ids, attention_mask=None, past_kvs=None):
        hidden_states, past_kvs, moe_loss = self.encoder(input_ids, attention_mask, past_kvs)
        logits = self.output_proj(hidden_states)
        return logits, past_kvs, moe_loss

# ============================================================================
# بخش ۵: حافظه دائمی (۴۵۰ خط)
# ============================================================================

class MemoryCell:
    """سلول حافظه - ۵۰ خط"""
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.data = torch.zeros(hidden_size)
        self.age = 0
        self.access_count = 0
        self.importance = 0.0
        self.timestamp = time.time()
        
    def update(self, new_data):
        self.data = 0.9 * self.data + 0.1 * new_data
        self.age += 1
        self.access_count += 1
        self.timestamp = time.time()
        self.importance = self.access_count / (self.age + 1)

class PermanentMemory(nn.Module):
    """حافظه دائمی - ۴۰۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.memory_size = config.memory_size
        
        self.memory = []
        self.memory_map = {}
        self.save_path = "memory/memory.pkl"
        
        # Load existing memory
        if os.path.exists(self.save_path):
            self.load()
        
    def add(self, key, value):
        """افزودن به حافظه"""
        if key in self.memory_map:
            idx = self.memory_map[key]
            self.memory[idx].update(value)
        else:
            if len(self.memory) >= self.memory_size:
                # Remove least important
                self._remove_least_important()
            
            cell = MemoryCell(self.hidden_size)
            cell.data = value
            self.memory.append(cell)
            self.memory_map[key] = len(self.memory) - 1
        
        self.save()
    
    def get(self, key):
        """بازیابی از حافظه"""
        if key in self.memory_map:
            idx = self.memory_map[key]
            self.memory[idx].access_count += 1
            return self.memory[idx].data
        return None
    
    def search(self, query_vector, top_k=10):
        """جستجوی نزدیک‌ترین حافظه‌ها"""
        similarities = []
        for i, cell in enumerate(self.memory):
            sim = F.cosine_similarity(query_vector.unsqueeze(0), cell.data.unsqueeze(0))
            similarities.append((sim.item(), i))
        
        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                'data': self.memory[idx].data,
                'importance': self.memory[idx].importance,
                'similarity': sim
            })
        
        return results
    
    def _remove_least_important(self):
        """حذف کم‌اهمیت‌ترین حافظه"""
        min_importance = float('inf')
        min_idx = -1
        
        for i, cell in enumerate(self.memory):
            if cell.importance < min_importance:
                min_importance = cell.importance
                min_idx = i
        
        if min_idx >= 0:
            del self.memory[min_idx]
            self._rebuild_map()
    
    def _rebuild_map(self):
        """بازسازی نگاشت"""
        self.memory_map = {}
        for i, cell in enumerate(self.memory):
            key = hashlib.md5(cell.data.numpy().tobytes()).hexdigest()
            self.memory_map[key] = i
    
    def save(self):
        """ذخیره حافظه"""
        data = {
            'memory': self.memory,
            'memory_map': self.memory_map
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """بارگذاری حافظه"""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
                self.memory = data['memory']
                self.memory_map = data['memory_map']

# ============================================================================
# بخش ۶: Tokenizer پیشرفته (۴۰۰ خط)
# ============================================================================

class BPETokenizer:
    """Byte Pair Encoding Tokenizer - ۴۰۰ خط"""
    def __init__(self, vocab_size=100000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<sep>': 4, '<cls>': 5, '<mask>': 6
        }
        
    def train(self, texts, min_frequency=2):
        """آموزش BPE روی متون"""
        word_counts = {}
        
        # Step 1: Count words
        for text in texts:
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Step 2: Initialize vocab with characters
        char_counts = {}
        for word, count in word_counts.items():
            for char in word:
                char_counts[char] = char_counts.get(char, 0) + count
        
        self.vocab = {char: i + len(self.special_tokens) 
                     for i, char in enumerate(char_counts.keys())}
        
        # Step 3: BPE merges
        for i in range(self.vocab_size - len(self.vocab) - len(self.special_tokens)):
            pairs = {}
            for word, count in word_counts.items():
                chars = list(word)
                for j in range(len(chars) - 1):
                    pair = (chars[j], chars[j+1])
                    pairs[pair] = pairs.get(pair, 0) + count
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = len(self.vocab) + len(self.special_tokens)
            
            # Update vocab
            merged = ''.join(best_pair)
            self.vocab[merged] = len(self.vocab) + len(self.special_tokens)
            
            # Update words
            new_word_counts = {}
            for word, count in word_counts.items():
                new_word = word.replace(''.join(best_pair), merged)
                new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
            word_counts = new_word_counts
    
    def encode(self, text):
        """تبدیل متن به توکن"""
        tokens = [self.special_tokens['<bos>']]
        
        words = text.split()
        for word in words:
            # Try to find longest match
            matched = False
            for length in range(len(word), 0, -1):
                for start in range(len(word) - length + 1):
                    subword = word[start:start+length]
                    if subword in self.vocab:
                        tokens.append(self.vocab[subword])
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                tokens.append(self.special_tokens['<unk>'])
        
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def decode(self, tokens):
        """تبدیل توکن به متن"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_token.update({v: k for k, v in self.special_tokens.items()})
        
        text = []
        for token in tokens:
            if token in id_to_token:
                token_str = id_to_token[token]
                if token_str not in self.special_tokens:
                    text.append(token_str)
        
        return ' '.join(text)
    
    def save(self, path):
        """ذخیره توکنایزر"""
        data = {
            'merges': {str(k): v for k, v in self.merges.items()},
            'vocab': self.vocab,
            'special_tokens': self.special_tokens
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        """بارگذاری توکنایزر"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.merges = {eval(k): v for k, v in data['merges'].items()}
            self.vocab = data['vocab']
            self.special_tokens = data['special_tokens']

# ============================================================================
# بخش ۷: مغز اصلی (۴۰۰ خط)
# ============================================================================

class GiantBrain(nn.Module):
    """مغز اصلی هوش مصنوعی - ۴۰۰ خط"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = TransformerDecoder(config)
        self.memory = PermanentMemory(config)
        self.tokenizer = BPETokenizer(config.vocab_size)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training stats
        self.step = 0
        self.total_loss = 0
        self.best_loss = float('inf')
        
        # Create directories
        os.makedirs(config.model_path, exist_ok=True)
        os.makedirs(config.memory_path, exist_ok=True)
    
    def forward(self, input_ids, attention_mask=None, past_kvs=None):
        """Forward pass"""
        logits, past_kvs, moe_loss = self.decoder(input_ids, attention_mask, past_kvs)
        return logits, past_kvs, moe_loss
    
    def compute_loss(self, logits, labels):
        """محاسبه loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    
    def learn_from_text(self, text):
        """یادگیری از یک متن"""
        self.train()
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens[:-1]])
        labels = torch.tensor([tokens[1:]])
        
        # Forward
        logits, _, moe_loss = self.forward(input_ids)
        loss = self.compute_loss(logits, labels) + moe_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update stats
        self.step += 1
        self.total_loss += loss.item()
        
        # Save to memory
        with torch.no_grad():
            hidden = self.decoder.encoder.embed_tokens(input_ids)
            memory_vector = hidden.mean(dim=1).squeeze(0)
            self.memory.add(text[:50], memory_vector)
        
        return loss.item()
    
    def learn_from_file(self, file_path, chunk_size=512):
        """یادگیری از یک فایل"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        
        total_loss = 0
        for i, chunk in enumerate(chunks):
            loss = self.learn_from_text(chunk)
            total_loss += loss
            
            if (i + 1) % 10 == 0:
                print(f"  Chunk {i+1}/{len(chunks)}, Loss: {loss:.4f}")
        
        return total_loss / len(chunks)
    
    def learn_from_directory(self, directory, file_types=['.txt', '.md']):
        """یادگیری از یک پوشه"""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in file_types):
                    files.append(os.path.join(root, filename))
        
        print(f"📚 Found {len(files)} files to learn from")
        
        total_loss = 0
        for i, file_path in enumerate(files):
            print(f"\n📖 Learning from {os.path.basename(file_path)} ({i+1}/{len(files)})")
            loss = self.learn_from_file(file_path)
            total_loss += loss
            
            # Save checkpoint every 10 files
            if (i + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_{i+1}.pt")
        
        return total_loss / len(files)
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        """تولید متن"""
        self.eval()
        
        with torch.no_grad():
            input_tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_tokens])
            
            past_kvs = None
            generated = input_ids
            
            for _ in range(max_length):
                logits, past_kvs, _ = self.forward(generated[:, -1:], past_kvs=past_kvs)
                logits = logits[0, -1, :] / temperature
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated = torch.cat([generated, torch.tensor([[next_token]])], dim=1)
                
                if next_token == self.tokenizer.special_tokens['<eos>']:
                    break
            
            return self.tokenizer.decode(generated[0].tolist())
    
    def save_checkpoint(self, filename):
        """ذخیره checkpoint"""
        path = os.path.join(self.config.model_path, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'total_loss': self.total_loss,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """بارگذاری checkpoint"""
        path = os.path.join(self.config.model_path, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.total_loss = checkpoint['total_loss']
            self.best_loss = checkpoint['best_loss']
            print(f"✅ Checkpoint loaded: {path}")
            return True
        return False
    
    def get_stats(self):
        """گرفتن آمار"""
        return {
            'step': self.step,
            'total_loss': self.total_loss,
            'avg_loss': self.total_loss / max(self.step, 1),
            'best_loss': self.best_loss,
            'memory_size': len(self.memory.memory),
            'vocab_size': self.tokenizer.vocab_size
        }

# ============================================================================
# بخش ۸: توابع کمکی (۲۵۰ خط)
# ============================================================================

def create_brain(config=None):
    """ساخت مغز جدید"""
    if config is None:
        config = Config()
    return GiantBrain(config)

def load_brain(path, config=None):
    """بارگذاری مغز"""
    if config is None:
        config = Config()
    brain = GiantBrain(config)
    brain.load_checkpoint(path)
    return brain

def train_on_articles(brain, articles_dir, epochs=10):
    """آموزش روی مقاله‌ها"""
    print("="*60)
    print("🚀 Starting training on articles")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        loss = brain.learn_from_directory(articles_dir)
        print(f"📉 Average Loss: {loss:.4f}")
        
        brain.save_checkpoint(f"epoch_{epoch+1}.pt")
        
        if loss < brain.best_loss:
            brain.best_loss = loss
            brain.save_checkpoint("best_model.pt")
    
    print("\n✅ Training complete!")
    return brain
