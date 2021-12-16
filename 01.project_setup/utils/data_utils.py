def normalize_inputs(data):
    """
    Normalizes the inputs to [-1, 1]

    :param data: input data array
    :return: normalized data to [-1, 1]
    """
    data = (data - 127.5) / 127.5
    return data


def get_noise(batch_size,n_noise):
    return tf.random.normal([batch_size,n_noise])



import tarfile
def extract(tar_url, extract_path='.'):
    print(tar_url)
    tar = tarfile.open(tar_url+'/flower_photos.tgz', 'r:gz')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])