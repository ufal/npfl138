#!/usr/bin/env python3
import argparse
import os

import timm
import torch
import transformers

import npfl138
npfl138.require_version("2425.13")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--allow_tf32", default=1, type=int, help="Allow TF32.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--compile", default=False, action="store_true", help="Compile the model.")
parser.add_argument("--ddp", default=False, action="store_true", help="Use DistributedDataParallel.")
parser.add_argument("--dp", default=False, action="store_true", help="Use DataParallel.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--epoch_batches", default=100, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--model", default="robeczech", type=str, help="Default model.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)

    # Create a RoBeCzech or EfficientNetV2B0 model for benchmarking.
    if args.model == "robeczech":
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = transformers.RobertaModel.from_pretrained("ufal/robeczech-base")
                self.model.pooler = None

            def forward(self, x):
                attention_mask = torch.ones_like(x, dtype=torch.float32)
                return self.model(x, attention_mask=attention_mask).last_hidden_state.mean(dim=1)

            def random_input(self, batch_size):
                return torch.randint(self.model.get_input_embeddings().num_embeddings, (1, 256)).tile(batch_size, 1)

            def random_target(self, batch_size):
                return torch.rand(768).tile(batch_size, 1)

    elif args.model == "efficientnet":
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

            def forward(self, x):
                return self.model(x)

            def random_input(self, batch_size):
                return torch.rand(1, 3, 224, 224).tile(batch_size, 1, 1, 1)

            def random_target(self, batch_size):
                return torch.rand(1, 1280).tile(batch_size, 1)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Compute the batch size for the current training, starting with `args.batch_size.`
    batch_size = args.batch_size

    # To use DDP, several processes must be started. One possibility is to use `torch.distributed.run` module:
    #   python3 -m torch.distributed.run --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 multi_gpu_training_example.py --ddp
    # which starts 4 concurrent processes in a common process group.

    # When using DDP, initialize the process group and update the per-device batch size.
    if args.ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")
        device_id = torch.distributed.get_rank() % torch.cuda.device_count()
        batch_size = batch_size // torch.distributed.get_world_size()

    # Create the chosen model.
    model = Model()

    # Create the training dataloader with random data.
    train_size = args.epoch_batches * batch_size
    train = torch.utils.data.StackDataset(model.random_input(train_size), model.random_target(train_size))
    train = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=args.dataloader_workers, persistent_workers=args.dataloader_workers > 0)

    # Optional model compilation.
    if args.compile:
        model.compile()

    # Use DataParallel if requested.
    if args.dp:
        model = torch.nn.DataParallel(model)

    # Use DistributedDataParallel if requested.
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model.to(device_id), device_ids=[device_id])
        print(f"Started a DDP process on device {device_id}.")

    # Train
    trainer = npfl138.TrainableModule(model)
    trainer.configure(
        optimizer=torch.optim.Adam(trainer.parameters()),
        loss=torch.nn.MSELoss(),
    )
    trainer.fit(train, epochs=args.epochs)

    # Correctly shut down the process group backend.
    if args.ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
