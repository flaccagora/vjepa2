

if __name__ == "__main__":
    import torch
    from transformers import AutoModel, AutoProcessor
    from transformers.image_utils import load_image

    # load the model and processor
    ckpt = "google/siglip2-base-patch16-512"
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    # load the image
    image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
    inputs = processor(images=[image], return_tensors="pt").to(model.device)

    # run infernece
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)    

    print(image_embeddings.shape)
