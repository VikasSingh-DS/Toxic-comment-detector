import transformers

# define the max lenght we going to use in tokenizing
MAX_LEN = 320

# define the train batch size
TRAIN_BATCH_SIZE = 14

# define the validation batch size
VALID_BATCH_SIZE = 7

# Number of epochs
EPOCHS = 3

# define distil base uncased path
DISTIL_BERT_PATH = (
    r"D:/My Workspace/Toxic-comment-detector/input/distilbert-base-uncased"
)

# define the saving model path
MODEL_PATH = r"D:/My Workspace/Toxic-comment-detector/weight.bin"

# define training file path
TRAINING_FILE = (
    r"D:/My Workspace/Toxic-comment-detector/input/jigsaw-toxic-comment-train.csv"
)

# Load the  DISTIL_BERT tokenizer
TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(
    DISTIL_BERT_PATH, do_lower_case=True
)
