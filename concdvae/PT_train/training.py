import time
import torch
import torch.distributed as dist
import sys
import pandas as pd
best_loss = 10000000


def train(cfg, model, datamodule, optimizer, scheduler, hydra_dir, best_loss_old):
    global best_loss
    train_loss_epoch = []
    trainpre_loss_epoch = []
    val_loss_epoch = []
    num_atom_accuracy_epoch = []
    lengths_mard_epoch = []
    angles_mae_epoch = []
    volumes_mard_epoch = []
    type_accuracy_epoch = []
    predict_mses_epoch = []
    if best_loss_old != None:
        best_loss = best_loss_old

    for epoch in range(cfg.train.PT_train.start_epochs, cfg.train.PT_train.max_epochs):
        sys.stdout.flush()

        train_loss,trainpre_loss = train_step(cfg, model, datamodule.train_dataloader, optimizer, epoch)

        val_losses, num_atom_accuracys, lengths_mards, angles_maes, volumes_mards, type_accuracys, predict_mses\
            = val_step(cfg, model, datamodule.val_dataloaders, optimizer, epoch)

        scheduler.step(metrics=val_losses[0].avg)


        if cfg.accelerator == 'DDP':
            loss = torch.tensor(val_losses[0].avg).cuda()
            torch.cuda.synchronize()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if torch.distributed.get_rank() == 0:
                world_size = torch.cuda.device_count()
                loss_avg = loss / world_size
                if (loss_avg > best_loss):
                    ##save model every epoch
                    filename = 'model_' + cfg.expname+'_notbest' + '.pth'
                    path = hydra_dir / filename
                    data = {'model': model.module.state_dict(),
                            'epoch': epoch + 1,
                            'val_loss': loss_avg}
                    torch.save(data, path)

                if (loss_avg < best_loss):
                    best_loss = loss_avg
                    filename = 'model_' + cfg.expname + '.pth'
                    path = hydra_dir / filename
                    data = {'model': model.module.state_dict(),
                            'epoch': epoch + 1,
                            'val_loss': loss_avg}
                    torch.save(data, path)
                    print('save model with loss = ', loss_avg, file=sys.stdout)

                    loss_dict = {
                        'train_loss_epoch': train_loss_epoch,
                        'val_loss_epoch': val_loss_epoch,
                        'num_atom_accuracy_epoch' : num_atom_accuracy_epoch,
                        'lengths_mard_epoch' : lengths_mard_epoch,
                        'angles_mae_epoch' : angles_mae_epoch,
                        'volumes_mard_epoch' : volumes_mard_epoch,
                        'type_accuracy_epoch' : type_accuracy_epoch,
                        'predict_mses_epoch' : predict_mses_epoch,
                        'trainpre_loss_epoch': trainpre_loss_epoch,
                    }
                    loss_df = pd.DataFrame(loss_dict)
                    excel_file = hydra_dir / 'loss_file.xlsx'
                    loss_df.to_excel(excel_file, index=False)

        else:
            if(val_losses[0].avg < best_loss):
                best_loss = val_losses[0].avg
                filename = 'model_' + cfg.expname + '.pth'
                path = hydra_dir / filename
                data = {'model': model.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': best_loss}
                torch.save(data, path)
                print('save model with loss = ', best_loss, file=sys.stdout)

                loss_dict = {
                    'train_loss_epoch': train_loss_epoch,
                    'val_loss_epoch': val_loss_epoch,
                    'num_atom_accuracy_epoch' : num_atom_accuracy_epoch,
                    'lengths_mard_epoch' : lengths_mard_epoch,
                    'angles_mae_epoch' : angles_mae_epoch,
                    'volumes_mard_epoch' : volumes_mard_epoch,
                    'type_accuracy_epoch' : type_accuracy_epoch,
                    'predict_mses_epoch' : predict_mses_epoch,
                    'trainpre_loss_epoch': trainpre_loss_epoch,
                }
                loss_df = pd.DataFrame(loss_dict)
                excel_file = hydra_dir / 'loss_file.xlsx'
                loss_df.to_excel(excel_file, index=False)
            
            else:
                filename = 'model_' + cfg.expname+'_notbest' + '.pth'
                path = hydra_dir / filename
                data = {'model': model.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': val_losses[0].avg}
                torch.save(data, path)

        train_loss_epoch.append(train_loss.avg.cpu().detach().numpy())
        trainpre_loss_epoch.append(trainpre_loss.avg.cpu().detach().numpy())
        val_loss_epoch.append(val_losses[0].avg.cpu().detach().numpy())
        num_atom_accuracy_epoch.append(num_atom_accuracys[0].avg.cpu().detach().numpy())
        lengths_mard_epoch.append(lengths_mards[0].avg.cpu().detach().numpy())
        angles_mae_epoch.append(angles_maes[0].avg.cpu().detach().numpy())
        volumes_mard_epoch.append(volumes_mards[0].avg.cpu().detach().numpy())
        type_accuracy_epoch.append(type_accuracys[0].avg.cpu().detach().numpy())
        predict_mses_epoch.append(predict_mses[0].avg.cpu().detach().numpy())


    test_losses = val_step(cfg, model, datamodule.val_dataloaders, optimizer, epoch, prefix='test')

    # output loss
    if cfg.accelerator == 'DDP':
        if torch.distributed.get_rank() == 0:
            # output_csvlog(loss_dict,hydra_dir)
            loss_dict = {
                'train_loss_epoch': train_loss_epoch,
                'val_loss_epoch': val_loss_epoch,
                'num_atom_accuracy_epoch' : num_atom_accuracy_epoch,
                'lengths_mard_epoch' : lengths_mard_epoch,
                'angles_mae_epoch' : angles_mae_epoch,
                'volumes_mard_epoch' : volumes_mard_epoch,
                'type_accuracy_epoch' : type_accuracy_epoch,
                'predict_mses_epoch' : predict_mses_epoch,
                'trainpre_loss_epoch': trainpre_loss_epoch,
            }
            loss_df = pd.DataFrame(loss_dict)
            excel_file = hydra_dir / 'loss_file.xlsx'
            loss_df.to_excel(excel_file, index=False)
    else:
        # output_csvlog(loss_dict,hydra_dir)
        loss_dict = {
            'train_loss_epoch': train_loss_epoch,
            'val_loss_epoch': val_loss_epoch,
            'num_atom_accuracy_epoch': num_atom_accuracy_epoch,
            'lengths_mard_epoch': lengths_mard_epoch,
            'angles_mae_epoch': angles_mae_epoch,
            'volumes_mard_epoch': volumes_mard_epoch,
            'type_accuracy_epoch': type_accuracy_epoch,
            'predict_mses_epoch': predict_mses_epoch,
            'trainpre_loss_epoch': trainpre_loss_epoch,
        }
        loss_df = pd.DataFrame(loss_dict)
        excel_file = hydra_dir / 'loss_file.xlsx'
        loss_df.to_excel(excel_file, index=False)

    return test_losses, train_loss_epoch, val_loss_epoch


def train_step(cfg, model, train_loader, optimizer, epoch):
    num_atom_loss = AverageMeter()
    lattice_loss = AverageMeter()
    coord_loss = AverageMeter()
    type_loss = AverageMeter()
    kld_loss = AverageMeter()
    composition_loss = AverageMeter()
    # property_loss = AverageMeter()
    train_loss = AverageMeter()
    trainpre_loss = AverageMeter()

    batch_time = AverageMeter()


    teacher_forcing = (epoch <= cfg.model.teacher_forcing_max_epoch)
    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):

        if cfg.accelerator != 'cpu':
            batch = batch.cuda()
        outputs = model(batch, teacher_forcing=teacher_forcing, training=True)
        if cfg.accelerator == 'DDP':
            log_dict, loss = model.module.compute_stats(batch, outputs, prefix='train')
        else:
            log_dict, loss = model.compute_stats(batch, outputs, prefix='train')


        optimizer.zero_grad()
        loss.backward()

        if cfg.train.PT_train.clip_grad_norm > 0:
            if(epoch<cfg.train.PT_train.clip_grad_norm_epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.PT_train.clip_grad_norm)
        optimizer.step()

        loss.data.cpu()
        num_atom_loss.update(outputs['num_atom_loss'].data.cpu(), outputs['z'].size(0))
        lattice_loss.update(outputs['lattice_loss'].data.cpu(), outputs['z'].size(0))
        coord_loss.update(outputs['coord_loss'].data.cpu(), outputs['z'].size(0))
        type_loss.update(outputs['type_loss'].data.cpu(), outputs['z'].size(0))
        kld_loss.update(outputs['kld_loss'].data.cpu(), outputs['z'].size(0))
        composition_loss.update(outputs['composition_loss'].data.cpu(), outputs['z'].size(0))
        train_loss.update(loss, outputs['z'].size(0))
        trainpre_loss.update(outputs['predict_loss'].data.cpu(), outputs['z'].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.train.PT_train.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time}\t'
                'Loss {train_loss}'.format(epoch, i, len(train_loader),
                                            batch_time=batch_time,
                                            train_loss=train_loss), file=sys.stdout)
            sys.stdout.flush()
        sys.stdout.flush()  

    return train_loss,trainpre_loss


def val_step(cfg, model, val_loaders, optimizer, epoch, prefix='val'):
    # switch to evaluate mode
    model.eval()

    val_losses = []
    num_atom_accuracys = []
    lengths_mards = []
    angles_maes = []
    volumes_mards = []
    type_accuracys = []
    predict_mses = []
    for val_loader in val_loaders:
        val_loss = AverageMeter()
        num_atom_accuracy = AverageMeter()
        lengths_mard = AverageMeter()
        angles_mae = AverageMeter()
        volumes_mard = AverageMeter()
        type_accuracy = AverageMeter()
        predict_mse = AverageMeter()

        batch_time = AverageMeter()

        end = time.time()
        for i, batch in enumerate(val_loader):

            if cfg.accelerator != 'cpu':
                batch = batch.cuda()
            outputs = model(batch, teacher_forcing=False, training=True)
            if cfg.accelerator == 'DDP':
                log_dict, loss = model.module.compute_stats(batch, outputs, prefix=prefix)
            else:
                log_dict, loss = model.compute_stats(batch, outputs, prefix=prefix)

            loss.data.cpu()
            val_loss.update(log_dict[f'{prefix}_loss'].data.cpu(), outputs['z'].size(0))
            num_atom_accuracy.update(log_dict[f'{prefix}_natom_accuracy'].data.cpu(), outputs['z'].size(0))
            lengths_mard.update(log_dict[f'{prefix}_lengths_mard'].data.cpu(), outputs['z'].size(0))
            angles_mae.update(log_dict[f'{prefix}_angles_mae'].data.cpu(), outputs['z'].size(0))
            volumes_mard.update(log_dict[f'{prefix}_volumes_mard'].data.cpu(), outputs['z'].size(0))
            type_accuracy.update(log_dict[f'{prefix}_type_accuracy'].data.cpu(), outputs['z'].size(0))
            predict_mse.update(log_dict[f'{prefix}_predict_loss'].data.cpu(), outputs['z'].size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.train.PT_train.print_freq == 0:
                print('{3}: [{0}][{1}/{2}]\t'
                    'Time {batch_time}\t'
                    'Loss {val_loss}'.format(epoch, i, len(val_loader), prefix,
                                                batch_time=batch_time,
                                                val_loss=val_loss), file=sys.stdout)

        if prefix == 'test':
            print('-----------------Test Result------------------')
            print(f'{prefix}_loss', val_loss)
            print(f'{prefix}_natom_accuracy', num_atom_accuracy)
            print(f'{prefix}_lengths_mard', lengths_mard)
            print(f'{prefix}_angles_mae', angles_mae)
            print(f'{prefix}_volumes_mard', volumes_mard)
            print(f'{prefix}_type_accuracy', type_accuracy)
            print(f'{prefix}_predict_mse', predict_mse)

        val_losses.append(val_loss)
        num_atom_accuracys.append(num_atom_accuracy)
        lengths_mards.append(lengths_mard)
        angles_maes.append(angles_mae)
        volumes_mards.append(volumes_mard)
        type_accuracys.append(type_accuracy)
        predict_mses.append(predict_mse)

    if prefix == 'test':
        return val_losses
    else:
        return val_losses, num_atom_accuracys, lengths_mards, angles_maes, volumes_mards, type_accuracys, predict_mses

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return ('%.3f(%.3f)'%(self.val,self.avg))