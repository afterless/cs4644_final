import torch as t
import copy
import wandb
from utils.weight_matching import weight_matching, apply_permutation


def straight_through_estimator(ps, modelA, modelB, train_loader, loss_fn, device, args):
    """
    Train two models to match each other's weights using straight-through estimation
    modelA and modelB are the two modules to be used
    training_data and test_data are the DataLoaders to be used
    args is the set of arguments to be used specified in each procedure
    """
    model = copy.deepcopy(modelA)
    pi_model = copy.deepcopy(modelB)
    model.to(device)
    pi_model.to(device)

    params_b = {k: v for k, v in modelB.named_parameters()}
    params_model = {k: v.detach() for k, v in model.named_parameters()}

    pi_model_state_dict = pi_model.state_dict()
    final_perm = None

    for p in model.parameters():  # might not be necessary
        p.requires_grad = True

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
    # wandb.init(project="perm_matching", config=args)
    # wandb.watch(model, log="all")
    for e in range(args.num_epochs):
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            perm = weight_matching(ps, params_model, params_b)
            proj_params = apply_permutation(ps, perm, params_b)

            for (
                (n, p_m),
                p_a,
                p_p,
            ) in zip(
                model.named_parameters(), modelA.parameters(), proj_params.values()
            ):
                pi_model_state_dict[n] = 0.5 * (p_a + ((p_p - p_m).detach() + p_m))

            optimizer.zero_grad()
            output = t.func.functional_call(pi_model, pi_model_state_dict, data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            del params_model
            params_model = {k: v.detach() for k, v in model.named_parameters()}
            loss = loss.item()
            print("Train Loss: {:.6f}".format(loss))
            final_perm = perm
            # wandb.log({"train_loss": loss})

    return final_perm
