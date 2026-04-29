import json
import tensorflow as tf
from tqdm import tqdm
from parameters import para_config
from models.transceiver import Transeiver, Mine
from dataset.dataloader import return_loader
from utlis.trainer import train_step, eval_step
from utlis.tools import SeqtoText, SNR_to_noise


def _dataset_cardinality(ds):
    try:
        card = tf.data.experimental.cardinality(ds).numpy()
    except Exception:
        return None
    if card < 0:
        return None
    return int(card)

if __name__ == '__main__':
    # Set random seed
    tf.random.set_seed(5)
    # Set Parameters
    args = para_config()
    # Load the vocab
    vocab = json.load(open(args.vocab_path, 'rb'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, args.end_idx)
    # Load dataset
    train_dataset, test_dataset = return_loader(args)
    train_total = _dataset_cardinality(train_dataset)
    test_total = _dataset_cardinality(test_dataset)
    # Build Model
    mine_net = Mine()
    net = Transeiver(args)
    # Define the optimizer
    optim_net = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
    optim_mi = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Training the model
    checkpoints = tf.train.Checkpoint(Transceiver=net)
    manager = tf.train.CheckpointManager(checkpoints, directory=args.checkpoint_path, max_to_keep=3)
    # Training the entire net
    best_loss = 10
    for epoch in range(args.epochs):
        n_std = SNR_to_noise(args.train_snr)
        train_loss_record, test_loss_record = 0, 0
        train_iter = tqdm(
            train_dataset,
            total=train_total,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
        )
        for (batch, (inp, tar)) in enumerate(train_iter):
            train_loss, train_loss_mine, _ = train_step(inp, tar, net, mine_net, optim_net, optim_mi, args.channel, n_std,
                                            train_with_mine=args.train_with_mine)
            train_loss_record += train_loss
            train_iter.set_postfix(loss=float(train_loss))
        train_loss_record = train_loss_record/batch

        # Valid
        test_iter = tqdm(
            test_dataset,
            total=test_total,
            desc=f"Epoch {epoch + 1}/{args.epochs} [test]",
            leave=False,
        )
        for (batch, (inp, tar)) in enumerate(test_iter):
            test_loss = eval_step(inp, tar, net, args.channel, n_std)
            test_loss_record += test_loss
            test_iter.set_postfix(loss=float(test_loss))
        test_loss_record = test_loss_record / batch

        if best_loss > test_loss_record:
            best_loss = test_loss_record
            manager.save(checkpoint_number=epoch)

        print('Epoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(epoch + 1, train_loss_record, test_loss_record))

