def greedy_decode(model, src_ord, src_res, pat_cov, max_len, BOS_idx, EOS_idx):
    '''
    autoregressively generate tokens using greedy decoding 
    '''
    model.eval()

    src_pat_emb = model.pat_cov_emb(pat_cov)
    src_pat_emb = src_pat_emb.unsqueeze(0).repeat(max_len, 1, 1)
    src_emb = torch.add(torch.mul(model.alpha_o, model.src_ord_emb(src_ord)), 
                        torch.mul(model.alpha_r, model.src_res_emb(src_res)))  
    src_emb = torch.add(src_emb, src_pat_emb) # (80, 1, 256)
    memory = model.transformer.encoder(src_emb) # (80, 1, 256)

    # Start the output tensor with the start symbol
    ys = torch.ones(1, 1).fill_(BOS_idx).type_as(src_ord.data)

    for i in range(max_len-1):
        #Decode one step at a time 
        tgt_emb = model.tgt_tok_emb(ys)
        out = model.transformer.decoder(tgt_emb, memory)
        out = out.transpose(0,1)
        #pass embeddings through generator to get out logits
        prob = model.generator(out[:, -1]) 
        _, next_word = torch.max(prob, dim=1)
        #convert to python int
        next_word = next_word.item() #convert to python int

        #Append the predicted word to the output sequence
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type_as(src_ord.data)], dim=0)

        #Break if EOS token is predicted
        if next_word == EOS_idx: 
            break
    return ys 