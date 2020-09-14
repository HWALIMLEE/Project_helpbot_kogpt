import torch
import torch.nn.functional as F
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    top_k = min(top_k, logits.size(-1))  # Safety check(10이 넘으면 안되니까)

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

        print('torch.topk(logits, top_k) : {}\n'.format(torch.topk(logits, top_k)))
        print('torch.topk(logits, top_k)[0] : {}\n'.format(torch.topk(logits, top_k)[0]))

        # ...은 이전 모든 축 고려(:을 여러번 쓰는 것과 동일)
        # None이 있으면 리스트의 축을 그대로 유지
        print('torch.topk(logits, top_k)[0][..., -1, None] : {}\n'.format(torch.topk(logits, top_k)[0][..., -1, None]))
        print('torch.topk(logits, top_k)[0][-1] : {}\n'.format(torch.topk(logits, top_k)[0][-1]))

        print('indices_to_remove : {}\n'.format(indices_to_remove))
        print('Top_K logits : {}\n'.format(logits))

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        print('sorted_logits : {}\n'.format(sorted_logits))
        print('sorted_indices : {}\n'.format(sorted_indices))

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        print('softmax : {}\n'.format(F.softmax(sorted_logits, dim=-1)))
        print('cumulative_probs : {}\n'.format(cumulative_probs))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        print('sorted_indices_to_remove : {}\n'.format(sorted_indices_to_remove))

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        print('sorted_indices_to_remove[..., 1:] : {}\n'.format(sorted_indices_to_remove))

        sorted_indices_to_remove[..., 0] = 0
        print('sorted_indices_to_remove[..., 0] : {}\n'.format(sorted_indices_to_remove))

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



# Here is how to use this function for top-p sampling
temperature = 1 # 0~1사이(0은 불가) / 1이면 원래 확률, 낮으면 확률을 큰 쪽에 몰아줌
top_k = 5
top_p = 0.7

# Get logits with a forward pass in our model (input is pre-defined)
logits = torch.tensor([[[0.1, 0.2, 0.3, 0.8, 0.6, 0.3, 0.5, 0.3, 0.4, 0.1], 
                        [0.2, 0.1, 0.7, 0.8, 0.5, 0.6, 0.8, 0.4, 0.3, 0.2]]])
print('logits : {}\n'.format(logits))

# Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
logits = logits[0, -1, :] / temperature
print('logits[0, -1, :] / temperature : {}\n'.format(logits))

filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
print('filtered_logits : {}\n'.format(filtered_logits))

# Sample from the filtered distribution
probabilities = F.softmax(filtered_logits, dim=-1)
print('probabilities : {}\n'.format(probabilities))

next_token = torch.multinomial(probabilities, 1)
print('next_token : {}'.format(next_token))

print('=======================================================================================')

# 온도를 1에서 0.1로 변경
# 마지막 probabilities가 4개에서 2개로 높은 쪽에 확률값을 몰아줌

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

        print('torch.topk(logits, top_k) : {}\n'.format(torch.topk(logits, top_k)))
        print('torch.topk(logits, top_k)[0] : {}\n'.format(torch.topk(logits, top_k)[0]))

        # ...은 이전 모든 축 고려(:을 여러번 쓰는 것과 동일)
        # None이 있으면 리스트의 축을 그대로 유지
        print('torch.topk(logits, top_k)[0][..., -1, None] : {}\n'.format(torch.topk(logits, top_k)[0][..., -1, None]))
        print('torch.topk(logits, top_k)[0][-1] : {}\n'.format(torch.topk(logits, top_k)[0][-1]))

        print('indices_to_remove : {}\n'.format(indices_to_remove))
        print('Top_K logits : {}\n'.format(logits))

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        print('sorted_logits : {}\n'.format(sorted_logits))
        print('sorted_indices : {}\n'.format(sorted_indices))

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        print('softmax : {}\n'.format(F.softmax(sorted_logits, dim=-1)))
        print('cumulative_probs : {}\n'.format(cumulative_probs))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        print('sorted_indices_to_remove : {}\n'.format(sorted_indices_to_remove))

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        print('sorted_indices_to_remove[..., 1:] : {}\n'.format(sorted_indices_to_remove))

        sorted_indices_to_remove[..., 0] = 0
        print('sorted_indices_to_remove[..., 0] : {}\n'.format(sorted_indices_to_remove))

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



# Here is how to use this function for top-p sampling
temperature = 0.1 # 0~1사이(0은 불가) / 1이면 원래 확률, 낮으면 확률을 큰 쪽에 몰아줌
top_k = 5
top_p = 0.7

# Get logits with a forward pass in our model (input is pre-defined)
logits = torch.tensor([[[0.1, 0.2, 0.3, 0.8, 0.6, 0.3, 0.5, 0.3, 0.4, 0.1], 
                        [0.2, 0.1, 0.7, 0.8, 0.5, 0.6, 0.8, 0.4, 0.3, 0.2]]])
print('logits : {}\n'.format(logits))

# Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
logits = logits[0, -1, :] / temperature
print('logits[0, -1, :] / temperature : {}\n'.format(logits))

filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
print('filtered_logits : {}\n'.format(filtered_logits))

# Sample from the filtered distribution
probabilities = F.softmax(filtered_logits, dim=-1)
print('probabilities : {}\n'.format(probabilities))

next_token = torch.multinomial(probabilities, 1)
print('next_token : {}'.format(next_token))