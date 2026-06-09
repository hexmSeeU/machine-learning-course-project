## Submission

Instructions and submission guidelines:

1. Submit your jupyter notebook.

## Task

In this assignment, you will build, train, and analyze a 6-layer decoder-only transformer model using PyTorch. The goal is to generate text in the style of Shakespearean poetry using a dataset of Shakespeare’s plays. This project involves both theoretical understanding and practical implementation of transformer models.

- Build a Transformer Model with detailed implementation of the following modules:
   \- Embedding layer
   \- Positional embedding (RoPE)
   \- Layer norm
   \- Masked attention
   \- FFN
- Compute the FLOPs and parameter count layer by layer (both theoretically and experimentally)
- Train an autoregressive model to generate poems of Shakespeare:
   \- TinyShakespeare dataset: https://www.kaggle.com/datasets/thedevastator/the-bards-best-a-character-modeling-dataset)
   \- Tokenize the text using llama2 tokenizer
   \- Visualize the training loss and the generated sample

## Deliverables

All the implementation details should be included in the jupyter notebook, including:

- Include all the training details
- Well-documented code
- Visualize the loss curve
- Justify the model size used to train on this dataset, as well as chosen hyper-parameters including batchsize, lr, etc.)

#### Evaluation Criteria

- **Code Quality and Organization**: Clear, readable, and well-organized code.
- **Model Implementation**: Accuracy in implementing the transformer architecture as per specifications.
- **Training Process**: Effective training process, with rational hyperparameter settings.
- **Analysis and Reporting**: Depth of analysis in the report, particularly in the discussion of model architecture, parameter efficiency, and training dynamics.