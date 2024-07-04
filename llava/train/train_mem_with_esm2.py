from llava.train.train_with_esm2 import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
