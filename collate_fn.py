import torch

def collate_fn(data):
    n_batch = len(data)
    data.sort(key=lambda x: len(x['txt']), reverse=True)

    txt_len = torch.tensor([len(x['txt']) for x in data])
    max_txt_len = max(txt_len)

    origin_txt = []
    txt = torch.zeros(n_batch, max_txt_len).long()

    gender = torch.zeros(n_batch).long()
    age = torch.zeros(n_batch).long()
    emotion = torch.zeros(n_batch).long()

    attributes_which = torch.zeros(n_batch, max_txt_len).int()
    attributes_how = torch.zeros(n_batch, max_txt_len).long()

    emb = torch.zeros((n_batch, 256))

    filename = []

    for ii, item in enumerate(data):
        origin_txt.append(item['origin_txt'])
        txt[ii, :len(item['txt'])] = torch.tensor(item['txt']).long()

        attributes_which[ii, :len(item['txt'])] = torch.tensor(item['attributes'][0]).int()
        attributes_how[ii, :len(item['txt'])] = torch.tensor(item['attributes'][1]).long()

        gender[ii]  = item['style']['gender']
        age[ii]     = item['style']['age']
        emotion[ii] = item['style']['emotion']
        filename.append(item['filename'])

    return origin_txt, txt, txt_len, attributes_which, attributes_how, gender, age, emotion, emb, filename
