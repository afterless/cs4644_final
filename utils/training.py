import torch.nn.functional as F
import torch as t
from tqdm import tqdm
import wandb


def train(model, device, train_loader, loss_fn, optimizer, epoch, args):
    model.train()
    correct = 0
    loss_total = 0
    t = 0
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)[:, -1]  # breaks encapsulation fml
        loss = loss_fn(output, target)
        pred = output.argmax(dim=-1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
            if wandb.run is not None:
                table = wandb.Table(
                    columns=[
                        "operands",
                        "sum",
                    ]
                )
                for data, pred in zip(data[:5], pred[:5]):
                    table.add_data(f"{data[0].item()}+{data[1].item()}", pred.item())
                wandb.log({"examples": table})

        loss_total += loss.item()
        t += 1

    acc = 100.0 * correct / len(train_loader.dataset)
    if wandb.run is not None:
        wandb.log(
            {
                "train_loss": loss_total / t,
                "train_acc": acc,
            }
        )

    print(
        "Train Accuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(train_loader.dataset), acc
        )
    )


def test(model, device, test_loader, loss):
    test_loss = 0
    correct = 0
    i = 0
    with t.inference_mode():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)[:, -1]  # breaks encapsulation fml
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=-1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            i += 1
    test_loss /= i
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "Average Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc
        )
    )
    if wandb.run is not None:
        wandb.log({"test_loss": test_loss, "test_acc": acc})
    return test_loss, acc
