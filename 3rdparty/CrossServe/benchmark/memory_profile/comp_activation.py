# Q: (bs, hxw / 16 / 16 + max_seq_len, head_num, head_size)
# K: (bs, hxw / 16 / 16 + max_seq_len, head_num, head_size)
# V: (bs, hxw / 16 / 16 + max_seq_len, head_num, head_size)
def compute_activation_size(bs, h, w, max_seq_len, head_num, head_size):
    "type: bf16"
    numels = bs * (h * w / 16 / 16 + max_seq_len) * head_num * head_size * 3
    return numels * 2 / 1024 / 1024 / 1024  # GB


print(compute_activation_size(32, 720, 720, 256, 24, 128))
print(compute_activation_size(16, 2048, 2048, 256, 24, 128))
print(compute_activation_size(4, 4096, 4096, 256, 24, 128))  # 4.51 GB for Q,K,V
