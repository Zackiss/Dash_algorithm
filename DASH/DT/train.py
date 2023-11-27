import os
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DASH.DT.data_loader import TrajectoryDataset, TrajectoryDataLoader
from offline_dt import DecisionTransformer


def DT_train(DT_filepath, state_dim, act_dim, KT_model, start_state_zero):
    save_model_name = "dt_model.pt"
    save_model_path = os.path.join("./", save_model_name)

    n_blocks = 3  # num of transformer blocks
    embed_dim = 128  # embedding (hidden) dim of transformer
    n_heads = 1  # num of transformer heads
    dropout_p = 0.1  # dropout probability
    device = torch.device("cuda")
    context_len = 20  # K in decision transformer

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    ).to(device)

    if not os.path.exists(DT_filepath):
        batch_size = 64  # training batch size
        lr = 1e-4  # learning rate
        wt_decay = 1e-4  # weight decay
        warmup_steps = 10000  # warmup steps for lr scheduler
        rtg_scale = 1000
        # total updates = max_train_iters x num_updates_per_iter
        max_train_iters = 200
        num_updates_per_iter = 100

        start_time = datetime.now().replace(microsecond=0)
        start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
        save_best_model_path = save_model_path[:-3] + "_best.pt"

        print("start time: " + start_time_str)
        print("=" * 60)

        print("device set to: " + str(device))
        print("model save path: " + save_model_path)

        traj_dataset = TrajectoryDataset(DT_filepath, context_len, rtg_scale)

        traj_data_loader = TrajectoryDataLoader(
            traj_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            kt_model=KT_model
        )

        data_iter = iter(traj_data_loader)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wt_decay
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        for i_train_iter in range(max_train_iters):

            log_action_losses = []
            model.train()

            for _ in range(num_updates_per_iter):
                try:
                    time_steps, states, actions, returns_to_go, traj_mask = next(data_iter)
                except StopIteration:
                    data_iter = iter(traj_data_loader)
                    time_steps, states, actions, returns_to_go, traj_mask = next(data_iter)

                time_steps = time_steps.to(device)  # B x T
                states = states.to(device)  # B x T x state_dim
                actions = actions.to(device)  # B x T x act_dim
                returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
                traj_mask = traj_mask.to(device)  # B x T
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = model.forward(
                    timesteps=time_steps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go
                )
                # only consider non padded elements
                action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
                action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())

            # save model
            print("saving current model at: " + save_model_path)
            torch.save(model.state_dict(), save_model_path)

        print("finished training!")
        print("=" * 60)
        end_time = datetime.now().replace(microsecond=0)
        time_elapsed = str(end_time - start_time)
        print("total training time: " + time_elapsed)
        print("saved max score model at: " + save_best_model_path)
        print("saved last updated model at: " + save_model_path)
        print("=" * 60)

    else:
        state_dict = torch.load(save_model_path)
        model.load_state_dict(state_dict)

    return model
